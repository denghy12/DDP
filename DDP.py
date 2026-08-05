import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from bce_loss import (
    build_ddp_classification_loss,
    two_way_prediction_diagnostics,
)
from build_cfg import setup_cfg
from evaluation_metrics import mAP, prf_cal
from models import ddp
from opts import arg_parser
from randaugment import RandAugment
from src.helper_functions.IncrementalDataset import build_dataset, build_loader
from src.helper_functions.detail_report import DetailReport
from src.helper_functions.emotic_loader import EMOTIC
from src.helper_functions.helper_functions import CutoutPIL


class DDP:
    def __init__(self):
        global args
        args = arg_parser().parse_args()
        self.set_seed(args.seed)
        self.cfg = setup_cfg(args)

        self.task_id = 0
        self.total_map = 0.0
        self.total_cf1 = 0.0
        self.total_of1 = 0.0
        self.base_classes = args.base_classes
        self.task_size = args.task_size
        self.total_classes = args.total_classes
        self._validate_protocol()

        self.dataset_name = args.dataset.lower()
        self.image_size = args.input_size
        self.num_epochs = args.epochs
        self.T_min = args.t_min
        self.T_max = args.t_max
        self.T_gamma = args.t_gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = build_ddp_classification_loss(
            args.ddp_classification_loss,
            gamma_neg=args.ddp_asl_gamma_neg,
            gamma_pos=args.ddp_asl_gamma_pos,
            clip=args.ddp_asl_clip,
            eps=args.ddp_asl_eps,
        )
        self.batch_size = args.train_batch_size or 8
        self.eval_batch_size = args.eval_batch_size or self.batch_size
        self.effective_batch_size = args.effective_batch_size or self.batch_size
        if self.effective_batch_size < self.batch_size:
            raise ValueError("effective_batch_size must be >= train_batch_size")
        self.accumulation_steps = math.ceil(
            self.effective_batch_size / self.batch_size
        )
        self.num_workers = args.num_workers
        self.run_name = args.name or (
            "emotic_upper_bound_ddp_semantic_threshold050"
            if self.dataset_name == "emotic" and args.upper_bound
            else "emotic_b5c3_ddp_semantic_tau2"
            if self.dataset_name == "emotic"
            else f"voc_b{self.base_classes}c{self.task_size}"
        )
        self.output_dir = os.path.abspath(args.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        project_root = os.path.dirname(os.path.abspath(__file__))
        if self.dataset_name == "emotic":
            self.root_dir = self.cfg.DATASET.ROOT or os.path.join(
                project_root, "datasets", "EMOTIC"
            )
            self._build_emotic_data()
        else:
            self.root_dir = self.cfg.DATASET.ROOT or os.path.join(
                project_root, "datasets", "VOC2007", "VOCdevkit", "VOC2007"
            )
            self._build_voc_data()

        self.model = ddp(self.cfg, self.classnames)
        self.model.to(self.device)
        self.schedule = [0, 20]
        self.optimizer_lr = args.lr or 5.9e-3
        self.optimizer = None
        self.scheduler = None
        self.training_history = []
        self.training_health = []
        self.build_optimizer_scheduler()
        self._write_main_loss_protocol()

        print(
            f"Dataset={self.dataset_name}, physical batch={self.batch_size}, "
            f"gradient accumulation={self.accumulation_steps}, "
            f"effective batch={self.batch_size * self.accumulation_steps}"
        )
        print(
            "DDP main classification loss="
            f"{args.ddp_classification_loss}, reduction=sum, "
            f"loss_w={args.loss_w}"
        )

    def _validate_protocol(self):
        if self.base_classes < 0:
            raise ValueError("base_classes must be >= 0")
        if self.task_size <= 0:
            raise ValueError("task_size must be > 0")
        if self.total_classes <= 0:
            raise ValueError("total_classes must be > 0")
        if self.base_classes > self.total_classes:
            raise ValueError("base_classes must not exceed total_classes")
        if self.base_classes == self.total_classes and not args.upper_bound:
            raise ValueError("base_classes must be smaller than total_classes")
        if args.upper_bound and self.base_classes != self.total_classes:
            raise ValueError(
                "upper_bound requires base_classes == total_classes"
            )
        if args.loss_w <= 0:
            raise ValueError("loss_w must be positive")
        if args.ddp_asl_gamma_neg < 0 or args.ddp_asl_gamma_pos < 0:
            raise ValueError("DDP ASL gamma values must be non-negative")
        if not 0 <= args.ddp_asl_clip < 1:
            raise ValueError("ddp_asl_clip must be in [0, 1)")
        if args.ddp_asl_eps <= 0:
            raise ValueError("ddp_asl_eps must be positive")

    def _incremental_stages(self):
        if args.upper_bound:
            stages = [(0, self.total_classes)]
            return stages[: args.max_tasks] if args.max_tasks is not None else stages

        stages = [(0, self.base_classes)] if self.base_classes > 0 else []
        start_class = self.base_classes if self.base_classes > 0 else 0
        stages.extend(
            (low, min(low + self.task_size, self.total_classes))
            for low in range(start_class, self.total_classes, self.task_size)
        )
        if args.max_tasks is not None:
            stages = stages[: args.max_tasks]
        return stages

    def _build_voc_data(self):
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                CutoutPIL(cutout_factor=0.2),
                RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        train_dataset = build_dataset(
            "voc",
            self.root_dir,
            0,
            self.total_classes,
            phase="train",
            transform=self.train_transforms,
        )
        self.classnames = train_dataset.CLASSES
        self.train_dataset_full = None
        self.val_dataset_full = None
        self.detail_report = None

    def _build_emotic_data(self):
        if args.upper_bound:
            if self.base_classes != 26 or self.total_classes != 26:
                raise ValueError(
                    "EMOTIC upper bound requires 26 joint classes"
                )
        elif (
            self.base_classes != 5
            or self.task_size != 3
            or self.total_classes != 26
        ):
            raise ValueError(
                "EMOTIC reproduction uses B5-C3: "
                "--base_classes 5 --task_size 3 --total_classes 26"
            )
        interpolation = transforms.InterpolationMode.BICUBIC
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=(0.05, 1.0),
                    ratio=(3.0 / 4.0, 4.0 / 3.0),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        resize_size = int((256 / 224) * self.image_size)
        self.val_transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
            ]
        )
        self.train_dataset_full = EMOTIC(
            self.root_dir,
            train=True,
            transform=self.train_transforms,
            input_mode=args.emotic_input_mode,
        )
        self.val_dataset_full = EMOTIC(
            self.root_dir,
            train=False,
            eval_splits=("val",),
            transform=self.val_transforms,
            input_mode=args.emotic_input_mode,
        )
        self.classnames = self.train_dataset_full.classes
        if len(self.classnames) != self.total_classes:
            raise RuntimeError(
                f"Expected {self.total_classes} EMOTIC classes, "
                f"found {len(self.classnames)}"
            )
        if args.upper_bound:
            class_mask = [list(range(self.total_classes))]
        else:
            class_mask = [list(range(0, self.base_classes))]
            class_mask.extend(
                list(range(low, min(low + self.task_size, self.total_classes)))
                for low in range(
                    self.base_classes, self.total_classes, self.task_size
                )
            )
        self._write_class_order(class_mask)
        self.detail_report = DetailReport(
            self.output_dir,
            self.run_name,
            self.classnames,
            class_mask,
            self.val_dataset_full.targets,
        )

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    @property
    def raw_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def build_optimizer_scheduler(self):
        model = self.raw_model
        self.optimizer = torch.optim.Adam(
            [
                {"params": model.prompt_learner.ctx_pos},
                {"params": model.prompt_learner.ctx_neg},
                {"params": model.visual_prompts},
            ],
            lr=self.optimizer_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.schedule, gamma=0.1
        )

    @staticmethod
    def _intersects(target, low_range, high_range):
        return any(low_range <= int(class_id) < high_range for class_id in target)

    def get_train_dataloader(self, low_range, high_range):
        if self.dataset_name == "emotic":
            indices = [
                index
                for index, target in enumerate(self.train_dataset_full.targets)
                if self._intersects(target, low_range, high_range)
            ]
            dataset = torch.utils.data.Subset(self.train_dataset_full, indices)
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        dataset = build_dataset(
            "voc",
            self.root_dir,
            low_range,
            high_range,
            phase="train",
            transform=self.train_transforms,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def get_val_seen_dataloader(self, high_range):
        if self.dataset_name == "emotic":
            indices = [
                index
                for index, target in enumerate(self.val_dataset_full.targets)
                if self._intersects(target, 0, high_range)
            ]
            dataset = torch.utils.data.Subset(self.val_dataset_full, indices)
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        dataset = build_dataset(
            "voc",
            self.root_dir,
            0,
            high_range,
            phase="val",
            transform=self.val_transforms,
        )
        return build_loader(
            dataset, self.eval_batch_size, self.num_workers, phase="val"
        )

    def _write_class_order(self, class_mask):
        detail_dir = os.path.join(self.output_dir, "detail")
        os.makedirs(detail_dir, exist_ok=True)
        order = []
        for task_id, class_ids in enumerate(class_mask):
            classes = [
                {
                    "class_id": int(class_id),
                    "class_name": self.classnames[class_id],
                }
                for class_id in class_ids
            ]
            order.append({"task": task_id, "classes": classes})
            print(
                f"[Class order] task {task_id}: "
                + ", ".join(
                    f"{item['class_id']}:{item['class_name']}" for item in classes
                )
            )
        path = os.path.join(detail_dir, f"{self.run_name}_class_order.json")
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(order, fp, ensure_ascii=False, indent=2)

    def _train_positive_counts(self):
        if self.train_dataset_full is None:
            return None
        counts = [0] * self.total_classes
        for target in self.train_dataset_full.targets:
            for class_id in target:
                counts[int(class_id)] += 1
        return {
            self.classnames[class_id]: int(count)
            for class_id, count in enumerate(counts)
        }

    def _main_loss_protocol(self):
        return {
            "ddp_main_classification_loss": args.ddp_classification_loss,
            "logit_pair_order": ["negative", "positive"],
            "binary_logit": "positive_logit - negative_logit",
            "reduction": "sum",
            "loss_w": float(args.loss_w),
            "asl": {
                "gamma_neg": float(args.ddp_asl_gamma_neg),
                "gamma_pos": float(args.ddp_asl_gamma_pos),
                "clip": float(args.ddp_asl_clip),
                "eps": float(args.ddp_asl_eps),
                "detach_focal_weight": True,
            },
            "supervision": "current_task_classes_only",
            "validation_role": "reporting_only",
            "training_time_eval_splits": ["val"],
            "test_used_during_training": False,
            "checkpoint_rule": "fixed_last_epoch",
            "epochs_per_task": int(self.num_epochs),
            "optimizer": "Adam",
            "learning_rate": float(self.optimizer_lr),
            "scheduler": {
                "name": "MultiStepLR",
                "milestones": list(self.schedule),
                "gamma": 0.1,
                "reset_each_task": bool(args.reset_optimizer_each_task),
            },
            "seed": int(args.seed),
            "base_classes": int(self.base_classes),
            "task_size": int(self.task_size),
            "total_classes": int(self.total_classes),
            "train_positive_counts": self._train_positive_counts(),
        }

    def _write_main_loss_protocol(self):
        path = os.path.join(self.output_dir, "ddp_main_loss_protocol.json")
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(
                self._main_loss_protocol(),
                fp,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

    def _write_training_diagnostics(self):
        path = os.path.join(self.output_dir, "training_diagnostics.json")
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "protocol": self._main_loss_protocol(),
                    "first_batch_health": self.training_health,
                    "epoch_history": self.training_history,
                },
                fp,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

    def _optimizer_gradient_norm(self, gradient_scale=1.0):
        gradient_scale = float(gradient_scale)
        if gradient_scale <= 0:
            gradient_scale = 1.0
        total_squared = 0.0
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach().float() / gradient_scale
                total_squared += float(gradient.square().sum().item())
        return math.sqrt(total_squared)

    def _try_record_training_health(
        self,
        pending_health,
        optimizer_step,
        accumulated_batches,
        gradient_scale,
    ):
        if pending_health is None:
            return None
        gradient_norm = self._optimizer_gradient_norm()
        if not math.isfinite(gradient_norm):
            pending_health["gradient_overflow_steps"] += 1
            print(
                "[Training health] non-finite scaled step skipped; "
                f"task={self.task_id}, optimizer_step={optimizer_step}, "
                f"grad_scale={gradient_scale}",
                flush=True,
            )
            return pending_health
        health_row = {
            **pending_health,
            "optimizer_step": int(optimizer_step),
            "gradient_accumulation_batches": int(accumulated_batches),
            "gradient_scale_before_step": float(gradient_scale),
            "prompt_gradient_norm": float(gradient_norm),
            "gradient_is_finite": True,
        }
        self.training_health.append(health_row)
        print(
            "[Training health] " + json.dumps(health_row, ensure_ascii=False),
            flush=True,
        )
        self._write_training_diagnostics()
        return None

    def _train_task(self, train_loader, low_range, high_range, scaler):
        self.model.train()
        pending_health = None
        optimizer_step = 0
        for epoch in range(self.num_epochs):
            print(f"epoch: {epoch}")
            self.optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            batch_count = 0
            supervised_elements = 0
            raw_loss_sum = 0.0
            weighted_loss_sum = 0.0
            for batch_id, (inputs, labels) in enumerate(train_loader):
                if (
                    args.max_train_batches is not None
                    and batch_id >= args.max_train_batches
                ):
                    break
                inputs = inputs.to(self.device, non_blocking=True).float()
                labels = labels.to(self.device, non_blocking=True).float()
                current_targets = labels[:, low_range:high_range]
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    outputs = self.model(
                        inputs, cls_id=(low_range, high_range), inference=False
                    )
                    raw_loss = self.criterion(outputs, current_targets)
                    loss = args.loss_w * raw_loss
                if epoch == 0 and batch_id == 0:
                    prediction_health = two_way_prediction_diagnostics(
                        outputs.detach(),
                        current_targets.detach(),
                        easy_negative_cutoff=args.ddp_asl_clip,
                    )
                    if hasattr(self.criterion, "diagnostics"):
                        prediction_health["asl"] = self.criterion.diagnostics(
                            outputs.detach(), current_targets.detach()
                        )
                    pending_health = {
                        "task": int(self.task_id),
                        "source_epoch": int(epoch),
                        "source_batch": int(batch_id),
                        "classes": [int(low_range), int(high_range)],
                        "raw_loss_sum": float(raw_loss.detach().item()),
                        "weighted_loss": float(loss.detach().item()),
                        "raw_loss_per_label": float(
                            raw_loss.detach().item() / current_targets.numel()
                        ),
                        "gradient_overflow_steps": 0,
                        **prediction_health,
                    }
                scaler.scale(loss).backward()
                accumulated += 1
                batch_count += 1
                supervised_elements += int(current_targets.numel())
                raw_loss_sum += float(raw_loss.detach().item())
                weighted_loss_sum += float(loss.detach().item())
                if accumulated == self.accumulation_steps:
                    gradient_scale = (
                        scaler.get_scale() if scaler.is_enabled() else 1.0
                    )
                    if scaler.is_enabled():
                        scaler.unscale_(self.optimizer)
                    pending_health = self._try_record_training_health(
                        pending_health,
                        optimizer_step,
                        accumulated,
                        gradient_scale,
                    )
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    accumulated = 0
                    optimizer_step += 1
            if accumulated:
                gradient_scale = scaler.get_scale() if scaler.is_enabled() else 1.0
                if scaler.is_enabled():
                    scaler.unscale_(self.optimizer)
                pending_health = self._try_record_training_health(
                    pending_health,
                    optimizer_step,
                    accumulated,
                    gradient_scale,
                )
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
            if self.scheduler is not None:
                self.scheduler.step()
            if supervised_elements <= 0:
                raise RuntimeError("No supervised labels were processed")
            epoch_row = {
                "task": int(self.task_id),
                "epoch": int(epoch),
                "batches": int(batch_count),
                "supervised_elements": int(supervised_elements),
                "raw_loss_sum": raw_loss_sum,
                "weighted_loss_sum": weighted_loss_sum,
                "raw_loss_per_label": raw_loss_sum / supervised_elements,
                "weighted_loss_per_label": (
                    weighted_loss_sum / supervised_elements
                ),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            }
            self.training_history.append(epoch_row)
            self._write_training_diagnostics()
            print(
                "[Epoch loss] " + json.dumps(epoch_row, ensure_ascii=False),
                flush=True,
            )
        if pending_health is not None:
            raise FloatingPointError(
                "No finite DDP prompt gradient was observed during the task"
            )

    def _temperature(self, high_range):
        if args.upper_bound:
            return 1.0
        denominator = self.total_classes - self.base_classes
        progress = (
            (high_range - self.base_classes) / denominator if denominator else 1.0
        )
        progress = max(0.0, min(1.0, progress))
        return self.T_min + (self.T_max - self.T_min) * math.pow(
            progress, self.T_gamma
        )

    def _evaluate_task(self, val_loader, high_range):
        self.model.eval()
        output_batches = []
        label_batches = []
        with torch.no_grad():
            for batch_id, (inputs, labels) in enumerate(val_loader):
                if (
                    args.max_eval_batches is not None
                    and batch_id >= args.max_eval_batches
                ):
                    break
                inputs = inputs.to(self.device, non_blocking=True).float()
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    outputs = self.model(
                        inputs, cls_id=(0, high_range), inference=True
                    )
                output_batches.append(outputs.detach().cpu())
                label_batches.append(labels.float().cpu())

        outputs_test = torch.cat(output_batches, dim=0)
        labels_full = torch.cat(label_batches, dim=0)
        labels_seen = labels_full[:, :high_range]
        temperature = self._temperature(high_range)
        pred = torch.softmax(outputs_test / temperature, dim=1)[:, 1, :]
        eval_loss = F.binary_cross_entropy(
            pred.clamp(1e-6, 1 - 1e-6), labels_seen, reduction="mean"
        ).item()

        map_score, _ = mAP(labels_seen.numpy(), pred.numpy())
        CP, CR, CF1, OP, OR, OF1 = prf_cal(
            pred, labels_seen, pred, threshold=args.thre
        )
        print(f"Task_id: {self.task_id}")
        print(
            f"Test: mAP={map_score:.4f}, CP={CP:.4f}, CR={CR:.4f}, "
            f"CF1={CF1:.4f}, OP={OP:.4f}, OR={OR:.4f}, OF1={OF1:.4f}, "
            f"T={temperature:.4f}"
        )

        self.total_map += map_score
        self.total_cf1 += CF1
        self.total_of1 += OF1
        if self.detail_report is not None:
            full_scores = torch.zeros(
                labels_full.shape[0], self.total_classes, dtype=pred.dtype
            )
            full_scores[:, :high_range] = pred
            overall = self.detail_report.update(
                self.task_id,
                full_scores,
                labels_full,
                args.thre,
                eval_loss,
            )
            print(
                f"[Average performances till task {self.task_id + 1}]"
                f"\tmAP: {overall['mAP']:.2f}"
                f"\tamAP: {overall['amAP']:.2f}"
                f"\toF1: {overall['oF1']:.2f}"
                f"\tcF1: {overall['cF1']:.2f}"
                f"\tLoss: {overall['loss']:.4f}"
            )

    def _save_checkpoint(self):
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        state = {
            "task": self.task_id,
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "args": vars(args),
            "classnames": self.classnames,
            "ddp_main_loss_protocol": self._main_loss_protocol(),
            "training_health": self.training_health,
        }
        torch.save(
            state, os.path.join(checkpoint_dir, f"task{self.task_id}.pth")
        )
        torch.save(state, os.path.join(self.output_dir, "checkpoints.pth"))

    def train_test(self):
        scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")
        stages = self._incremental_stages()
        print("Incremental stages:", stages)
        for low_range, high_range in stages:
            if args.reset_optimizer_each_task and self.task_id > 0:
                self.build_optimizer_scheduler()
            train_loader = self.get_train_dataloader(low_range, high_range)
            val_loader = self.get_val_seen_dataloader(high_range)
            print(
                f"🚀 Task {self.task_id} training: classes "
                f"[{low_range}, {high_range}), "
                f"train samples={len(train_loader.dataset)}, "
                f"eval samples={len(val_loader.dataset)}"
            )
            self._train_task(
                train_loader, low_range, high_range, scaler
            )
            print("✅ Training finished")
            self._evaluate_task(val_loader, high_range)
            print("✅ Test finished")
            self._save_checkpoint()
            self.task_id += 1
