"""
Module for converting labelbox.com JSON exports to MS COCO format.

https://github.com/Labelbox/labelbox-python/blob/master/labelbox/exporters/coco_exporter.py
"""

import datetime as dt
import json
import os
import logging
from typing import Any, Dict

from PIL import Image, ExifTags
import requests
from shapely import wkt
from shapely.geometry import Polygon

##temp
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches


def gen_next_id():
    n = 0
    while True:
        yield n
        n += 1


LOGGER = logging.getLogger(__name__)
ENCODING_DICT = dict()
ENCODING_GENERATOR = gen_next_id()


def from_json(labeled_data, coco_output, label_format='XY', download_data=False, data_path=''):
    """Writes labelbox JSON export into MS COCO format."""
    # read labelbox JSON output
    with open(os.path.expanduser(labeled_data), 'r') as file_handle:
        label_data = json.loads(file_handle.read())

    # setup COCO dataset container and info
    coco = make_coco_metadata(label_data[0]['Project Name'], label_data[0]['Created By'],)

    for data in label_data:
        # Download and get image name
        try:
            add_label(coco, data['ID'], data['Labeled Data'], data['Label'], label_format,
                      download_data, data_path)
        except requests.exceptions.MissingSchema as exc:
            LOGGER.warning(exc)
            continue
        except requests.exceptions.ConnectionError:
            LOGGER.warning('Failed to fetch image from %s, skipping', data['Labeled Data'])
            continue

    with open(os.path.expanduser(coco_output), 'w+') as file_handle:
        file_handle.write(json.dumps(coco))


def make_coco_metadata(project_name: str, created_by: str) -> Dict[str, Any]:
    """Initializes COCO export data structure.
    Args:
        project_name: name of the project
        created_by: email of the project creator
    Returns:
        The COCO export represented as a dictionary.
    """
    return {
        'info': {
            'year': dt.datetime.now(dt.timezone.utc).year,
            'version': None,
            'description': project_name,
            'contributor': created_by,
            'url': 'labelbox.com',
            'date_created': dt.datetime.now(dt.timezone.utc).isoformat()
        },
        'images': [],
        'annotations': [],
        'licenses': [],
        'categories': []
    }


def encode_label(label):
    if label in ENCODING_DICT:
        return ENCODING_DICT[label]
    else:
        next_id = next(ENCODING_GENERATOR)
        ENCODING_DICT[label] = next_id
        return next_id


def add_label(
        coco: Dict[str, Any], label_id: str, image_url: str,
        labels: Dict[str, Any], label_format: str, download_data=False, data_path=''):
    """Incrementally updates COCO export data structure with a new label.
    Args:
        coco: The current COCO export, will be incrementally updated by this method.
        label_id: ID for the instance to write
        image_url: URL to download image file from
        labels: Labelbox formatted labels to use for generating annotation
        label_format: Format of the labeled data. Valid options are: "WKT" and
                      "XY", default is "WKT".
        download_data: Should download data from internet or try to find it locally
        data_path: Path where to take image data from
    Returns:
        The updated COCO export represented as a dictionary.
    """
    if type(label_id) == str:
        label_id = encode_label(label_id)
    image = {
        "id": label_id,
        "license": None,
        "flickr_url": image_url,
        "coco_url": image_url,
        "date_captured": None,
    }

    if download_data:
        response = requests.get(image_url, stream=True, timeout=10.0)
        response.raw.decode_content = True
        image_raw = Image.open(response.raw)
    else:
        name = '/'.join(image['coco_url'].split('/')[3:])
        image["file_name"] = name
        try:
            #fix_orientation(os.path.expanduser(data_path + name))
            image_raw = Image.open(os.path.expanduser(data_path + name))
        except FileNotFoundError:
            return

    image['width'], image['height'] = image_raw.size

    # remove classification labels (Skip, etc...)
    if not callable(getattr(labels, 'keys', None)):
        return

    # convert label to COCO Polygon format
    for category_name, label_data in labels.items():
        try:
            # check if label category exists in 'categories' field
            category_id = [c['id']
                           for c in coco['categories']
                           if c['supercategory'] == category_name][0]
        except IndexError:
            category_id = len(coco['categories']) + 1
            category = {
                'supercategory': category_name,
                'id': category_id,
                'name': category_name
            }
            coco['categories'].append(category)

        polygons = _get_polygons(label_format, label_data)
        if polygons and len(polygons) > 0:
            _append_polygons_as_annotations(coco, image, category_id, polygons)
            coco['images'].append(image)
        else:
            continue

        #TEMP
        # img = cv2.imread(os.path.expanduser(data_path+name))
        # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #
        # fig, ax = plt.subplots(1)
        # for polygon in polygons:
        #     bounds = np.array([polygon.bounds[0], polygon.bounds[1],
        #              polygon.bounds[2] - polygon.bounds[0],
        #              polygon.bounds[3] - polygon.bounds[1]], dtype=int)
        #     width = bounds[2] - bounds[0]
        #     height = bounds[3] - bounds[1]
        #     bottom_left = (bounds[0], bounds[1])
        #     # Create a Rectangle patch
        #     # rect = patches.Rectangle(bottom_left, width, height, linewidth=1, edgecolor='r', facecolor='none')
        #     #
        #     # # Add the patch to the Axes
        #     # ax.add_patch(rect)
        #     ax.scatter(polygon.bounds[0], polygon.bounds[1])
        #     ax.scatter(polygon.bounds[0], polygon.bounds[3])
        #     ax.scatter(polygon.bounds[2], polygon.bounds[1])
        #     ax.scatter(polygon.bounds[2], polygon.bounds[3])
        #
        #     cv2.rectangle(img2, tuple(bounds[:-2]), tuple(bounds[2:]), (0, 255, 0), 2)
        #     plt.plot(*polygon.exterior.xy)
        #
        #
        #
        # ax.imshow(img)
        # plt.show()
        # print('s')
        # # cv2.imshow('test', img2)


def _append_polygons_as_annotations(coco, image, category_id, polygons):
    "Adds `polygons` as annotations in the `coco` export"
    for polygon in polygons:
        segmentation = []
        for x_val, y_val in polygon.exterior.coords:
            segmentation.extend([x_val, y_val])

        annotation = {
            "id": len(coco['annotations']) + 1,
            "image_id": image['id'],
            "category_id": category_id,
            "segmentation": [segmentation],
            "area": polygon.area,  # float
            "bbox": [polygon.bounds[0], polygon.bounds[1], polygon.bounds[2], polygon.bounds[3]],
                    # We keep Shirel's data format
                    # For COCO refer to this
                    #polygon.bounds[2] - polygon.bounds[0],
                    #polygon.bounds[3] - polygon.bounds[1]],
            "iscrowd": 0
        }

        coco['annotations'].append(annotation)


def _get_polygons(label_format, label_data):
    "Converts segmentation `label: String!` into polygons"
    if label_format == 'WKT':
        if isinstance(label_data, list):  # V3
            polygons = map(lambda x: wkt.loads(x['geometry']), label_data)
        else:  # V2
            polygons = wkt.loads(label_data)
    elif label_format == 'XY':
        polygons = []
        for xy_list in label_data:
            if 'geometry' in xy_list:  # V3
                xy_list = xy_list['geometry']

                # V2 and V3
                if not isinstance(xy_list, list):
                    LOGGER.warning('Could not get an point list to construct polygon, skipping')
                    continue
            else:  # V2, or non-list
                if not isinstance(xy_list, list) or not xy_list or 'x' not in xy_list[0]:
                    # skip non xy lists
                    LOGGER.warning('Could not get an point list to construct polygon, skipping')
                    continue

            if len(xy_list) > 2:  # need at least 3 points to make a polygon
                polygons.append(Polygon(map(lambda p: (p['x'], p['y']), xy_list)))
    else:
        exc = ValueError(label_format)
        raise exc

    return polygons


def fix_orientation(filepath):
    flag = False
    try:
        image = Image.open(filepath)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            flag = True
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            flag = True
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            flag = True
            image = image.rotate(90, expand=True)
        image.save(filepath)
        image.close()
        return flag
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        return False
