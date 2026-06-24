import copy
import torch
import torch.nn as nn
import numpy as np
import os
import math
import random
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
from src.helper_functions.coco_loader import COCOLoader, coco_ids_to_cats, coco_fake2real
from src.helper_functions.IncrementalDataset import build_dataset, build_loader
import torchvision.transforms as transforms
from src.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay, \
    reduce_tensor, AverageMeter
from randaugment import RandAugment
from torch.optim import lr_scheduler
from evaluation_metrics import prf_cal, mAP
from models import (ddp)
from build_cfg import setup_cfg
from opts import arg_parser
from bce_loss import BCELoss
from utils import ReDirectSTD

class DDP:
    def __init__(self):
        global args
        parser = arg_parser()
        args = parser.parse_args()
        self.set_seed(args.seed)
        self.cfg = setup_cfg(args)

        self.task_id = 0
        self.total_map = 0
        self.total_cf1 = 0
        self.total_of1 = 0

        "VOC"
        self.base_classes = args.base_classes
        self.task_size = args.task_size
        self.total_classes = args.total_classes
        if self.base_classes < 0:
            raise ValueError("base_classes must be >= 0")
        if self.task_size <= 0:
            raise ValueError("task_size must be > 0")
        if self.total_classes <= 0:
            raise ValueError("total_classes must be > 0")
        if self.base_classes >= self.total_classes:
            raise ValueError("base_classes must be smaller than total_classes")
        self.dataset_name = "voc"
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = self.cfg.DATASET.ROOT or os.path.join(
            project_root, "datasets", "VOC2007", "VOCdevkit", "VOC2007"
        )
        "VOC"

        self.image_size = 224
        self.start = 0
        self.end = self.base_classes
        self.num_epochs = args.epochs
        self.weight_decay = 0.0
        self.momentum = 0.9
        self.schedule = [0, 20]
        self.schedule_type = 'decay'
        self.T_min = args.t_min
        self.T_max = args.t_max
        self.T_gamma = args.t_gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = BCELoss()

        self.train_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            CutoutPIL(cutout_factor=0.2),
            RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dataset_without_old = build_dataset(self.dataset_name, self.root_dir, 0, self.total_classes,
                                                  phase='train', transform=self.train_transforms)
        train_dataset = train_dataset_without_old
        self.classnames = train_dataset.CLASSES
        self.model = ddp(self.cfg, self.classnames)
        self.model.to(self.device)

        self.batch_size = args.train_batch_size or 8
        self.num_workers = args.num_workers
        self.base_lr = 0.009
        self.now_lr = 0.009
        self.lr = 0
        self.optimizer_lr = args.lr or 5.9e-3
        self.optimizer = None
        self.scheduler = None
        self.build_optimizer_scheduler()

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def build_optimizer_scheduler(self):
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.prompt_learner.ctx_pos},
                {"params": self.model.prompt_learner.ctx_neg},
                {"params": self.model.visual_prompts},
            ],
            lr=self.optimizer_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)

    def get_train_dataloader(self, low_range, high_range):
        train_dataset_without_old = build_dataset(self.dataset_name, self.root_dir, low_range, high_range,
                                                  phase='train', transform=self.train_transforms)
        train_dataset = train_dataset_without_old
        '''VOC'''
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                   num_workers=self.num_workers, drop_last=True)
        '''VOC'''
        return train_loader

    def get_val_seen_dataloader(self, high_range):
        val_dataset_seen = build_dataset(self.dataset_name, self.root_dir, 0, high_range, phase='val',
                                         transform=self.val_transforms)
        val_loader_seen = build_loader(val_dataset_seen, self.batch_size, self.num_workers, phase='val')
        return val_loader_seen

    def train_test(self):
        scaler = torch.cuda.amp.GradScaler()
        if self.base_classes > 0:
            incremental_stages = [(0, self.base_classes)]
            start_class = self.base_classes
        else:
            incremental_stages = []
            start_class = 0

        incremental_stages += [
            (low, min(low + self.task_size, self.total_classes))
            for low in range(start_class, self.total_classes, self.task_size)
        ]

        print("Incremental stages:", incremental_stages)
        for low_range, high_range in incremental_stages:
            if args.reset_optimizer_each_task:
                self.build_optimizer_scheduler()
            if low_range == 0:
                self.lr = self.base_lr
            else:
                self.lr = self.now_lr
            train_loader = self.get_train_dataloader(low_range, high_range)
            val_loader_seen = self.get_val_seen_dataloader(high_range)
            print(f"🚀 Training start: ")
            self.model.train()
            print('Task_id: ', self.task_id)
            for epoch in range(self.num_epochs):
                print('epoch: ', epoch)
                for i, (inputs, labels) in enumerate(train_loader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    inputs = inputs.float()
                    labels = labels.float()
                    labels_tr = labels.clone()
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    # forward
                    with torch.set_grad_enabled(True):
                        with torch.cuda.amp.autocast():
                            cls_id = (low_range, high_range)
                            outputs = self.model(inputs, cls_id=cls_id, inference=False)
                            loss = args.loss_w * self.criterion(outputs, labels_tr[:, low_range: high_range])

                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()

            print(f"\n✅ Training finished！")

            if self.task_id == self.task_id:
                self.model.eval()
                print(f"🚀 Test start: ")
                for i, (inputs, labels) in enumerate(val_loader_seen):
                    if inputs.size(0) > 1:
                        labels = labels[:, :high_range]
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        inputs = inputs.float()
                        labels = labels.float()
                        self.optimizer.zero_grad()

                        with torch.no_grad():
                            with torch.cuda.amp.autocast():
                                cls_id = (0, high_range)
                                outputs = self.model(inputs, cls_id=cls_id, inference=True) 
                                if i == 0:
                                    outputs_test = outputs
                                    labels_test = labels
                                else:
                                    outputs_test = torch.cat((outputs_test, outputs), 0)
                                    labels_test = torch.cat((labels_test, labels), 0)

                if outputs_test.dim() == 3:
                    # PCD
                    C_max = self.total_classes
                    C_t = high_range
                    C_init = self.base_classes
                    progress = (C_t - C_init) / (C_max - C_init)
                    self.T = self.T_min + (self.T_max - self.T_min) * math.pow(progress, self.T_gamma)
                    softmax_outputs = torch.softmax(outputs_test.detach().cpu() / self.T, dim=1)
                    pred = softmax_outputs[:, 1, :]
                else:
                    pred = torch.sigmoid(outputs_test.detach())

                mAP_score, _ = mAP(labels_test.to(torch.device("cpu")).numpy(), pred.to(torch.device("cpu")).numpy())
                print('Task_id: ', self.task_id)
                print('Test:')
                print('mAP', mAP_score)
                self.total_map = self.total_map + mAP_score
                print("mean_map", self.total_map / (self.task_id + 1))
                CP, CR, CF1, OP, OR, OF1 = prf_cal(pred.to(torch.device("cpu")), labels_test.to(torch.device("cpu")), pred.to(torch.device("cpu")))
                self.total_cf1 = self.total_cf1 + CF1
                self.total_of1 = self.total_of1 + OF1
                print('CP', CP)
                print('CR', CR)
                print('CF1', CF1)
                print('OP', OP)
                print('OR', OR)
                print('OF1', OF1)
                print(f"\n✅ Test finished！")

            # Create directories if they don't exist
            checkpoint_dir = (
                    './checkpoint/'
                    + "b" + str(self.base_classes)
                    + 'c' + str(self.task_size) + "/"
                    + "epoch" + str(self.num_epochs) + "/"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save model
            torch.save(self.model.state_dict(), checkpoint_dir + "task" + str(self.task_id) + '.pth')
            self.task_id = self.task_id + 1
            self.start = high_range
            self.end = high_range + self.task_size
