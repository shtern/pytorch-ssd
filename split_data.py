import json
import pathlib
import os
import numpy as np
import vision.utils.labelbox_to_coco as l2c

JSON_EXTENSION = '.json'
KEY_ID = 'id'
KEY_IMAGE_ID = 'image_id'
KEY_IMAGES = 'images'
KEY_ANNOTATIONS = 'annotations'


def relabel_ids(start):
    n = start
    while True:
        yield n
        n += 1


def write_split(source, images, annotations, path_to_json, dataset_type):
    name = os.path.splitext(path_to_json)[0]  # getting filename only
    name += '_' + dataset_type + JSON_EXTENSION
    with open(pathlib.Path(os.path.expanduser(name)), 'w+') as file:
        final_dict = source.copy()
        final_dict[KEY_IMAGES] = images
        final_dict[KEY_ANNOTATIONS] = annotations
        json.dump(final_dict, file)


def split_train_val_test(path_to_json, ratio):
    """
    Split dataset annotations to train, validation and test
    :param path_to_json: path to annotations file
    :param ratio: train/validation/test ratio
    """
    with open(pathlib.Path(os.path.expanduser(path_to_json))) as file:
        source = json.load(file)
        image_ids = list(set(item[KEY_ID] for item in source[KEY_IMAGES]))
        indices = np.array(image_ids)
        inds_split = np.random.multinomial(n=1, pvals=ratio, size=len(image_ids)).argmax(axis=1)

        train_inds = indices[inds_split == 0]
        val_inds = indices[inds_split == 1]
        test_inds = indices[inds_split == 2]

        for inds, dataset_type in zip([train_inds, val_inds, test_inds], ['train', 'validation', 'test']):
            images = []
            all_annotations = []
            for idx in inds:
                image = next(item for item in source[KEY_IMAGES] if item[KEY_ID] == idx)
                annotations = list(filter(lambda a: a[KEY_IMAGE_ID] == idx, source[KEY_ANNOTATIONS]))
                images.append(image)
                all_annotations.extend(annotations)
            write_split(source, images, all_annotations, path_to_json, dataset_type)


def merge_annotations(output, *args):
    if len(args) <= 1:
        raise ValueError('not enough files passed')

    start_image_id = 0
    start_annotation_id = 0
    encode_dict = dict()
    final = None
    for path in args:
        with open(pathlib.Path(os.path.expanduser(path))) as file:
            raw_json = json.load(file)
            if not final:
                final = raw_json
                start_image_id = max(image['id'] for image in final['images'])
                start_annotation_id = max(annotation['id'] for annotation in final['annotations'])
                relabel_images = relabel_ids(start_image_id+1)
                relabel_annotations = relabel_ids(start_annotation_id+1)
            else:
                for image in raw_json['images']:
                    new_id = next(relabel_images)
                    encode_dict[image['id']] = new_id
                    image['id'] = new_id
                for annotation in raw_json['annotations']:
                    annotation['id'] = next(relabel_annotations)
                    annotation['image_id'] = encode_dict.get(annotation['image_id'], annotation['image_id'])

                final['images'].extend(raw_json['images'])
                final['annotations'].extend(raw_json['annotations'])

    with open(pathlib.Path(os.path.expanduser(output)), 'w+') as file:
        json.dump(final, file)


def test():
    l2c.from_json('~/data/more_wounds/annotations/coco_annotations_lblbx.json',
                  '~/data/more_wounds/annotations/coco_annotations_train.json',
                  data_path='~/data/more_wounds/images/')
    merge_annotations('merged_annotations.json',
                      '~/data/wounds_dataset/annotations/coco_annotations.json',
                      '~/data/more_wounds/annotations/coco_annotations_train.json')
test()

