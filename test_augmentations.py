import cv2
import argparse
from vision.ssd.config import mobilenetv1_ssd_config
from vision.datasets.coco_dataset import CocoDataset
from vision.ssd.data_preprocessing import TrainAugmentation
from vision.transforms.transforms import *


class TestCocoDataset(CocoDataset):
    def get_for_aug_test(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['img_path'])
        boxes = image_info['boxes']
        labels = image_info['labels']

        orig_img = image
        orig_boxes = boxes

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)

        back_transforms = Compose([ToCV2Image(), ToAbsoluteCoords()])
        image, boxes, labels = back_transforms(image, boxes, labels)
        return image_info['image_id'], image, boxes, labels, orig_img, orig_boxes


def main():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Training With Pytorch')

    parser.add_argument('--dataset_path', nargs='+', help='Dataset directory path')

    args = parser.parse_args()

    config = mobilenetv1_ssd_config
    aug = TrainAugmentation(config.image_size, config.image_mean, config.image_std)

    dataset = TestCocoDataset(args.dataset_path[0],
                          transform=aug, target_transform=None,
                          dataset_type="train", balance_data=False)

    for i in range(len(dataset)):
        _, img, boxes, _, orig_img, orig_boxes = dataset.get_for_aug_test(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        for geom in boxes:
            cv2.rectangle(img, tuple(geom[:-2]), tuple(geom[2:]), (0, 255, 0), 6)
        cv2.namedWindow('augmented', cv2.WINDOW_NORMAL)
        cv2.imshow('augmented', img)
        cv2.resizeWindow('augmented', 600, 600)

        for geom in orig_boxes:
            cv2.rectangle(orig_img, tuple(geom[:-2]), tuple(geom[2:]), (0, 255, 0), 6)
        cv2.namedWindow('original', cv2.WINDOW_NORMAL)
        cv2.imshow('original', orig_img)
        cv2.resizeWindow('original', 600, 600)
        if cv2.waitKey(3000) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
