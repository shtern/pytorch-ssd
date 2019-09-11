import json
import pathlib
import os
import numpy as np

JSON_EXTENSION = '.json'
KEY_ID = 'id'
KEY_IMAGE_ID = 'image_id'
KEY_IMAGES = 'images'
KEY_ANNOTATIONS = 'annotations'


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



