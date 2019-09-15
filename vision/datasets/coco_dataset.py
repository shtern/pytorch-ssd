from pycocotools.coco import COCO
import numpy as np
import pathlib
import cv2
import os
import vision.utils.labelbox_to_coco as labelbox_to_coco

COCO_COVERTED_PATH = 'annotations_converted_coco.json'


class CocoDataset:
    def __init__(self, root, size_multiplication=1,
                 transform=None, target_transform=None,
                 dataset_type="train", balance_data=False):
        self.dataset_type = dataset_type.lower()
        self.size_multiplication = size_multiplication
        self.root = pathlib.Path(os.path.expanduser(root))
        annotation_file_path = f"{self.root}/annotations/coco_annotations_{self.dataset_type}.json"
        try:
            self.coco = COCO(annotation_file_path)
        except AssertionError:
            labelbox_to_coco.from_json(annotation_file_path, COCO_COVERTED_PATH)
            self.coco = COCO(COCO_COVERTED_PATH)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.anns_ids = list(sorted(self.coco.anns.keys()))
        self.transform = transform
        self.target_transform = target_transform

        self.data, self.class_names, self.class_dict = self._read_data()
        self.balance_data = balance_data
        self.min_image_num = -1
        if self.balance_data and self.size_multiplication == 1:
            self.data = self._balance_data()

        self.class_stat = None

    def _getitem(self, index):
        if index >= len(self.data):
            index = np.random.randint(0, index % len(self.data) + 1)
        image_info = self.data[index]
        image = self._read_image(image_info['img_path'])
        boxes = image_info['boxes']
        labels = image_info['labels']
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image_info['image_id'], image, boxes, labels

    def __getitem__(self, index):
        _, image, boxes, labels = self._getitem(index)
        return image, boxes, labels

    def get_annotation(self, index):
        """To conform the eval_ssd implementation that is based on the VOC dataset."""
        image_id, image, boxes, labels = self._getitem(index)
        is_difficult = np.zeros(boxes.shape[0], dtype=np.uint8)
        return image_id, (boxes, labels, is_difficult)

    def get_image(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['img_path'])
        if self.transform:
            image, _ = self.transform(image)
        return image

    def _read_data(self):
        big_cats_dict = self.coco.loadCats(self.coco.getCatIds())
        class_names = [d['name'] for d in big_cats_dict]
        class_dict = {class_name: i for i, class_name in enumerate(class_names)}

        pre_data = dict()

        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)
            pre_data[img_id] = {
                'image_id': img_id,
                'img_path': img_info[0].get('file_name', ''),
                'boxes': None,
                'labels': None
            }

        for ann_id in self.anns_ids:
            annotation = self.coco.loadAnns(ann_id)[0]
            img_id = annotation['image_id']
            box = np.array(annotation['bbox']).astype(np.float32)
            if type(pre_data[img_id]['boxes']) == np.ndarray:
                pre_data[img_id]['boxes'] = np.vstack((pre_data[img_id]['boxes'], box))
            else:
                pre_data[img_id]['boxes'] = np.array([box])

            if type(pre_data[img_id]['labels']) == np.ndarray:
                pre_data[img_id]['labels'] = np.append(pre_data[img_id]['labels'], annotation['category_id'])
            else:
                pre_data[img_id]['labels'] = np.array([annotation['category_id']])

        return list(pre_data.values()), class_names, class_dict

    def __len__(self):
        return self.size_multiplication * len(self.data)

    def __repr__(self):
        if self.class_stat is None:
            self.class_stat = {name: 0 for name in self.class_names[1:]}
            for example in self.data:
                for class_index in example['labels']:
                    class_name = self.class_names[class_index]
                    self.class_stat[class_name] += 1
        content = ["Dataset Summary:"
                   f"Number of Images: {len(self.data)}",
                   f"Minimum Number of Images for a Class: {self.min_image_num}",
                   "Label Distribution:"]
        for class_name, num in self.class_stat.items():
            content.append(f"\t{class_name}: {num}")
        return "\n".join(content)

    def _read_image(self, image_path_short):
        image_file = self.root / f"images/{image_path_short}"
        image = cv2.imread(str(image_file))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _balance_data(self):
        if self.size_multiplication > 1:
            raise ValueError('Cannot balance data while having size_multiplication > 1')
        label_image_indexes = [set() for _ in range(len(self.class_names))]
        for i, image in enumerate(self.data):
            for label_id in image['labels']:
                label_image_indexes[label_id].add(i)
        label_stat = [len(s) for s in label_image_indexes]
        self.min_image_num = min(label_stat[1:])
        sample_image_indexes = set()
        for image_indexes in label_image_indexes[1:]:
            image_indexes = np.array(list(image_indexes))
            sub = np.random.permutation(image_indexes)[:self.min_image_num]
            sample_image_indexes.update(sub)
        sample_data = [self.data[i] for i in sample_image_indexes]
        return sample_data


