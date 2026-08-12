import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

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
class CocoERTransforms:
    """Source-shaped three-view preprocessing plus audited head boxes."""

    transform: Any
    head_boxes: Mapping[str, Tuple[float, float, float, float]]
    train: bool = False
    image_size: int = 224


@dataclass(frozen=True)
class CocoERGPUTransforms:
    """Raw RGB transport for method-side GPU three-view preprocessing."""

    head_boxes: Mapping[str, Tuple[float, float, float, float]]
    train: bool = False
    image_size: int = 224


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

        if input_mode not in ("full", "person_crop", "body_context", "cocoer_multilevel"):
            raise ValueError(
                f"Invalid EMOTIC input_mode '{input_mode}'. "
                "Expected 'full', 'person_crop', 'body_context', or "
                "'cocoer_multilevel'."
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
        self.sample_keys = []

        split_items = []
        for split in self.splits:
            split_items.extend((split, item) for item in self._as_list(annotation_mat[split]))

        included_set = set(self.included_cats)
        occurrences = defaultdict(int)
        for split, item in split_items:
            img_path = os.path.join(
                self.path, "cvpr_emotic", item.folder, item.filename
            )
            if not os.path.isfile(img_path):
                raise RuntimeError(f"EMOTIC image not found at {img_path}")

            relative_path = os.path.relpath(img_path, self.path).replace(os.sep, "/")
            for person in self._as_list(item.person):
                ordinal = occurrences[(split, relative_path)]
                occurrences[(split, relative_path)] += 1
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
                self.sample_keys.append(
                    f"emotic:{split}:{relative_path}:person={ordinal}"
                )

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
        if self.input_mode == "cocoer_multilevel":
            if not isinstance(self.transform, (CocoERTransforms, CocoERGPUTransforms)):
                raise RuntimeError("CocoER requires a registered CocoER transform")
            sample_key = self.sample_keys[idx]
            if sample_key not in self.transform.head_boxes:
                raise KeyError(f"CocoER head-box cache is missing {sample_key}")
            body_box = self._valid_bbox(image, self.body_bboxes[idx], "body")
            head_box = self._valid_bbox(
                image, self.transform.head_boxes[sample_key], "head"
            )
            if isinstance(self.transform, CocoERGPUTransforms):
                # Decode exactly once on CPU. Cropping, stochastic augmentation,
                # resize, normalization, and geometry conversion occur as a
                # method-side batch operation on the selected CUDA device.
                image = torch.from_numpy(
                    np.array(image, dtype=np.uint8, copy=True)
                ).permute(2, 0, 1).contiguous()
                geometry = torch.tensor(
                    [body_box, head_box], dtype=torch.float32
                )
            else:
                image, body_box, head_box = self._cocoer_joint_augment(
                    image, body_box, head_box, self.transform.train
                )
                width, height = image.size
                body = image.crop(tuple(body_box))
                head = image.crop(tuple(head_box))
                context_tensor = self.transform.transform(image)
                body_tensor = self.transform.transform(body)
                head_tensor = self.transform.transform(head)
                expected = (3, self.transform.image_size, self.transform.image_size)
                if any(tuple(value.shape) != expected for value in (
                    context_tensor, body_tensor, head_tensor
                )):
                    raise ValueError("CocoER transforms must return 3x224x224 tensors")
                image = torch.stack((context_tensor, body_tensor, head_tensor), dim=0)
                scale_x = self.transform.image_size / float(width)
                scale_y = self.transform.image_size / float(height)
                geometry = torch.tensor(
                    [
                        [body_box[0] * scale_x, body_box[1] * scale_y,
                         body_box[2] * scale_x, body_box[3] * scale_y],
                        [head_box[0] * scale_x, head_box[1] * scale_y,
                         head_box[2] * scale_x, head_box[3] * scale_y],
                    ],
                    dtype=torch.float32,
                )
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
        if self.input_mode == "cocoer_multilevel":
            return image, target, geometry
        return image, target

    @classmethod
    def _valid_bbox(cls, image, bbox, name):
        values = np.asarray(bbox, dtype=np.float32).ravel()
        if values.size < 4 or not np.isfinite(values[:4]).all():
            raise ValueError(f"Invalid CocoER {name} bounding box")
        width, height = image.size
        x1 = max(0, min(width, int(np.floor(values[0]))))
        y1 = max(0, min(height, int(np.floor(values[1]))))
        x2 = max(0, min(width, int(np.ceil(values[2]))))
        y2 = max(0, min(height, int(np.ceil(values[3]))))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Degenerate CocoER {name} bounding box")
        return [x1, y1, x2, y2]

    @staticmethod
    def _cocoer_joint_augment(image, body_box, head_box, train):
        """Port the released random_crop2 and coordinated horizontal flip."""

        if not train:
            return image, body_box, head_box
        width, height = image.size
        hx1, hy1, hx2, hy2 = head_box
        bx1, by1, bx2, by2 = body_box
        ux1 = random.randint(min(hx1, bx1), hx1)
        uy1 = random.randint(min(hy1, by1), hy1)
        ux2 = random.randint(hx2, max(hx2, bx2))
        uy2 = random.randint(hy2, max(hy2, by2))
        crop_x1 = random.randint(0, ux1)
        crop_y1 = random.randint(0, uy1)
        crop_x2 = random.randint(ux2, width)
        crop_y2 = random.randint(uy2, height)
        image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        body_box = [ux1 - crop_x1, uy1 - crop_y1, ux2 - crop_x1, uy2 - crop_y1]
        head_box = [hx1 - crop_x1, hy1 - crop_y1, hx2 - crop_x1, hy2 - crop_y1]
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            flipped_width = image.size[0]
            body_box = [flipped_width - body_box[2], body_box[1],
                        flipped_width - body_box[0], body_box[3]]
            head_box = [flipped_width - head_box[2], head_box[1],
                        flipped_width - head_box[0], head_box[3]]
        return image, body_box, head_box

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
