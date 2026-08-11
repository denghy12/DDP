import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.io as sio
import torch
from PIL import Image


@dataclass(frozen=True)
class BodyContextTransforms:
    """Separate native transforms stored in one fixed-size batch tensor."""

    context: Any
    body: Any
    context_size: int = 224
    body_size: int = 128


@dataclass(frozen=True)
class BENetViewsTransform:
    """Package BENet full/person/context views and target geometry.

    Channels 0:3 are the full scene, 3:6 the target-person crop, 6:9 the
    target-masked context, channel 9 the target-box mask, and channel 10 the
    valid-image mask.  Geometry is transported without exposing labels.
    """

    train: bool
    size: int = 512
    flip_probability: float = 0.5

    @staticmethod
    def _clamp_bbox(bbox, width, height):
        values = np.asarray(bbox, dtype=np.float32).ravel()
        if values.size < 4 or not np.isfinite(values[:4]).all():
            raise ValueError("BENet requires a finite target-person bbox")
        x1, y1, x2, y2 = values[:4]
        x1, x2 = sorted((max(0.0, min(float(width), float(x1))), max(0.0, min(float(width), float(x2)))))
        y1, y2 = sorted((max(0.0, min(float(height), float(y1))), max(0.0, min(float(height), float(y2)))))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("BENet target-person bbox has zero area")
        return x1, y1, x2, y2

    def __call__(self, image, bbox):
        import torchvision.transforms.functional as functional

        width, height = image.size
        x1, y1, x2, y2 = self._clamp_bbox(bbox, width, height)
        if self.train and random.random() < self.flip_probability:
            image = functional.hflip(image)
            x1, x2 = width - x2, width - x1

        person = image.crop((int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))))
        context = image.copy()
        # Official MaskAllSubjects removes the person from the context branch.
        context_array = np.asarray(context).copy()
        context_array[int(np.floor(y1)):int(np.ceil(y2)), int(np.floor(x1)):int(np.ceil(x2))] = 0
        context = Image.fromarray(context_array)

        def normalized(value):
            value = functional.resize(value, [self.size, self.size])
            value = functional.to_tensor(value)
            return functional.normalize(value, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        scene_tensor = normalized(image)
        person_tensor = normalized(person)
        context_tensor = normalized(context)
        sx1 = max(0, min(self.size - 1, int(np.floor(x1 * self.size / width))))
        sy1 = max(0, min(self.size - 1, int(np.floor(y1 * self.size / height))))
        sx2 = max(sx1 + 1, min(self.size, int(np.ceil(x2 * self.size / width))))
        sy2 = max(sy1 + 1, min(self.size, int(np.ceil(y2 * self.size / height))))
        box_mask = torch.zeros((self.size, self.size), dtype=torch.float32)
        box_mask[sy1:sy2, sx1:sx2] = 1.0
        valid = torch.ones_like(box_mask)
        return torch.cat(
            (scene_tensor, person_tensor, context_tensor, box_mask[None], valid[None]),
            dim=0,
        )


class EMOTIC(torch.utils.data.Dataset):
    """EMOTIC loader matching multi-lane-main's train/(val+test) protocol."""

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        eval_splits=("val", "test"),
        input_mode="full",
        included=None,
        class_names=None,
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        self.splits = ["train"] if train else list(eval_splits)
        self.path = self._resolve_dataset_path(self.root)
        self.input_mode = input_mode
        self.included_cats = list(included) if included is not None else []

        if input_mode not in ("full", "person_crop", "body_context", "benet_views"):
            raise ValueError(
                f"Invalid EMOTIC input_mode '{input_mode}'. "
                "Expected 'full', 'person_crop', 'body_context', or 'benet_views'."
            )

        annotation_path = os.path.join(self.path, "CVPR17_Annotations.mat")
        if not os.path.isfile(annotation_path):
            raise RuntimeError(f"EMOTIC annotations not found at {annotation_path}")

        annotation_mat = sio.loadmat(
            annotation_path, squeeze_me=True, struct_as_record=False
        )
        if class_names is None:
            category_names = set()
            for split in ("train", "val", "test"):
                for item in self._as_list(annotation_mat[split]):
                    for person in self._as_list(item.person):
                        category_names.update(self._categories_from_person(person))
            self.classes = sorted(category_names)
        else:
            self.classes = list(class_names)
            if len(self.classes) != len(set(self.classes)):
                raise ValueError("class_names must not contain duplicates")
        self.CLASSES = self.classes
        self.category_names = {i: name for i, name in enumerate(self.classes)}
        self.class2idx = {name: i for i, name in self.category_names.items()}
        self.category2name = dict(self.category_names)

        self.file_paths = []
        self.targets = []
        self.body_bboxes = []

        split_items = []
        for split in self.splits:
            split_items.extend(self._as_list(annotation_mat[split]))

        included_set = set(self.included_cats)
        for item in split_items:
            img_path = os.path.join(
                self.path, "cvpr_emotic", item.folder, item.filename
            )
            if not os.path.isfile(img_path):
                raise RuntimeError(f"EMOTIC image not found at {img_path}")

            for person in self._as_list(item.person):
                categories = self._categories_from_person(person)
                target = sorted(
                    {self.class2idx[c] for c in categories if c in self.class2idx}
                )
                if not target:
                    continue
                if included_set and not included_set.intersection(target):
                    continue

                self.file_paths.append(img_path)
                self.targets.append(target)
                self.body_bboxes.append(person.body_bbox)

    @staticmethod
    def _resolve_dataset_path(root):
        direct = os.path.join(root, "CVPR17_Annotations.mat")
        nested = os.path.join(root, "EMOTIC", "CVPR17_Annotations.mat")
        if os.path.isfile(direct):
            return root
        if os.path.isfile(nested):
            return os.path.join(root, "EMOTIC")
        return root

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert("RGB")
        if self.input_mode == "benet_views":
            if not isinstance(self.transform, BENetViewsTransform):
                raise RuntimeError("BENet views require BENetViewsTransform")
            image = self.transform(image, self.body_bboxes[idx])
        elif self.input_mode == "body_context":
            body = self._crop_person(image, self.body_bboxes[idx])
            if self.transform is None:
                raise RuntimeError("EMOTIC requires an image transform")
            # The native EMOT-Net interface consumes full-scene context first
            # and the annotated person crop second.  Keeping both views inside
            # one tensor preserves the benchmark TrainBatch/EvaluationBatch
            # boundary without exposing any additional labels or metadata.
            if isinstance(self.transform, BodyContextTransforms):
                context_tensor = self.transform.context(image)
                body_tensor = self.transform.body(body)
                if context_tensor.shape[-2:] != (
                    self.transform.context_size,
                    self.transform.context_size,
                ):
                    raise ValueError("Context transform returned an unexpected size")
                if body_tensor.shape[-2:] != (
                    self.transform.body_size,
                    self.transform.body_size,
                ):
                    raise ValueError("Body transform returned an unexpected size")
                pad = self.transform.context_size - self.transform.body_size
                if pad < 0:
                    raise ValueError("Body canvas cannot exceed context canvas")
                # Normalized zero is the per-channel dataset mean. Padding is
                # only a transport detail; EMOT-Net crops back to 128x128.
                body_tensor = torch.nn.functional.pad(
                    body_tensor,
                    (0, pad, 0, pad),
                    value=0.0,
                )
            else:
                context_tensor = self.transform(image)
                body_tensor = self.transform(body)
            image = torch.stack((context_tensor, body_tensor), dim=0)
        elif self.input_mode == "person_crop":
            image = self._crop_person(image, self.body_bboxes[idx])
            if self.transform is None:
                raise RuntimeError("EMOTIC requires an image transform")
            image = self.transform(image)
        else:
            if self.transform is None:
                raise RuntimeError("EMOTIC requires an image transform")
            image = self.transform(image)

        target = torch.zeros(len(self.classes), dtype=torch.float32)
        target[self.targets[idx]] = 1
        return image, target

    @staticmethod
    def _crop_person(image, bbox):
        try:
            bbox = np.asarray(bbox, dtype=np.float32).ravel()
        except (TypeError, ValueError):
            return image
        if bbox.size < 4 or not np.isfinite(bbox[:4]).all():
            return image

        width, height = image.size
        x1, y1, x2, y2 = bbox[:4]
        x1 = max(0.0, min(float(width), float(x1)))
        y1 = max(0.0, min(float(height), float(y1)))
        x2 = max(0.0, min(float(width), float(x2)))
        y2 = max(0.0, min(float(height), float(y2)))
        if x2 <= x1 or y2 <= y1:
            return image

        return image.crop(
            (
                int(np.floor(x1)),
                int(np.floor(y1)),
                int(np.ceil(x2)),
                int(np.ceil(y2)),
            )
        )

    @staticmethod
    def _as_list(value):
        if isinstance(value, np.ndarray):
            return value.ravel().tolist()
        return [value]

    def _str_list(self, value):
        if isinstance(value, str):
            return [value]
        if isinstance(value, np.ndarray):
            values = []
            for item in value.ravel().tolist():
                values.extend(self._str_list(item))
            return values
        if isinstance(value, (list, tuple)):
            values = []
            for item in value:
                values.extend(self._str_list(item))
            return values
        return []

    def _categories_from_person(self, person):
        categories = []
        for annotation in self._as_list(person.annotations_categories):
            if hasattr(annotation, "categories"):
                categories.extend(self._str_list(annotation.categories))
            else:
                categories.extend(self._str_list(annotation))
        return categories
