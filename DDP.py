import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from bce_loss import BCELoss
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
        self.criterion = BCELoss()
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
            "emotic_b5c3_ddp_semantic_tau2"
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
        self.build_optimizer_scheduler()

        print(
            f"Dataset={self.dataset_name}, physical batch={self.batch_size}, "
            f"gradient accumulation={self.accumulation_steps}, "
            f"effective batch={self.batch_size * self.accumulation_steps}"
        )

    def _validate_protocol(self):
        if self.base_classes < 0:
            raise ValueError("base_classes must be >= 0")
        if self.task_size <= 0:
            raise ValueError("task_size must be > 0")
        if self.total_classes <= 0:
            raise ValueError("total_classes must be > 0")
        if self.base_classes >= self.total_classes:
            raise ValueError("base_classes must be smaller than total_classes")

    def _incremental_stages(self):
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
        if (
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
            eval_splits=("val", "test"),
            transform=self.val_transforms,
            input_mode=args.emotic_input_mode,
        )
        self.classnames = self.train_dataset_full.classes
        if len(self.classnames) != self.total_classes:
            raise RuntimeError(
                f"Expected {self.total_classes} EMOTIC classes, "
                f"found {len(self.classnames)}"
            )
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

    def _train_task(self, train_loader, low_range, high_range, scaler):
        self.model.train()
        for epoch in range(self.num_epochs):
            print(f"epoch: {epoch}")
            self.optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            for batch_id, (inputs, labels) in enumerate(train_loader):
                if (
                    args.max_train_batches is not None
                    and batch_id >= args.max_train_batches
                ):
                    break
                inputs = inputs.to(self.device, non_blocking=True).float()
                labels = labels.to(self.device, non_blocking=True).float()
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    outputs = self.model(
                        inputs, cls_id=(low_range, high_range), inference=False
                    )
                    loss = args.loss_w * self.criterion(
                        outputs, labels[:, low_range:high_range]
                    )
                scaler.scale(loss).backward()
                accumulated += 1
                if accumulated == self.accumulation_steps:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    accumulated = 0
            if accumulated:
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()

    def _temperature(self, high_range):
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
        CP, CR, CF1, OP, OR, OF1 = prf_cal(pred, labels_seen, pred)
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
