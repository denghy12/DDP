import os

import numpy as np
import scipy.io as sio
import torch
from PIL import Image


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
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        self.splits = ["train"] if train else list(eval_splits)
        self.path = self._resolve_dataset_path(self.root)
        self.input_mode = input_mode
        self.included_cats = list(included) if included is not None else []

        if input_mode not in ("full", "person_crop"):
            raise ValueError(
                f"Invalid EMOTIC input_mode '{input_mode}'. "
                "Expected 'full' or 'person_crop'."
            )

        annotation_path = os.path.join(self.path, "CVPR17_Annotations.mat")
        if not os.path.isfile(annotation_path):
            raise RuntimeError(f"EMOTIC annotations not found at {annotation_path}")

        annotation_mat = sio.loadmat(
            annotation_path, squeeze_me=True, struct_as_record=False
        )
        category_names = set()
        for split in ("train", "val", "test"):
            for item in self._as_list(annotation_mat[split]):
                for person in self._as_list(item.person):
                    category_names.update(self._categories_from_person(person))

        self.classes = sorted(category_names)
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
        if self.input_mode == "person_crop":
            image = self._crop_person(image, self.body_bboxes[idx])
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

