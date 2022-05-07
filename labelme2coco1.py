# -*- coding:utf-8 -*-
"""
@author:胡亚雄
@file: labelme2coco1.py
@time: 2022/04/28
@description:
"""
import json
import os
import glob
import shutil

import numpy as np
from labelme import utils
from sklearn.model_selection import train_test_split

np.random.seed(40)

CLASS_NAME_TO_ID = {"alligator": 1, "longitude": 2, "transverse": 3, "block": 4, "other": 5}


class Labelme2Coco(object):
    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.image_id = 0
        self.annotation_id = 0

    def save_coco_json(self, instance, save_path):
        """
        保存为coco形式的json文件
        :param instance:
        :param save_path:
        :return:
        """
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False,
                  indent=2)  # set indent = 2, much more beautiful

    def to_coco(self, json_path_list):
        # sourcery skip: inline-immediately-returned-variable
        """
        由json文件构建coco
        :param json_path_list:
        :return:
        """
        self._init_categories()
        for json_path in json_path_list:
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            for shape in shapes:
                annotation = self._annotation(shape)
                self.annotations.append(annotation)
                self.annotation_id += 1
            self.image_id += 1
        instance = {'info': 'pavement crack dataset',
                    'license': ['license'],
                    'images': self.images,
                    'annotations': self.annotations,
                    'categories': self.categories}
        return instance

    def _init_categories(self):
        for k, v in CLASS_NAME_TO_ID.items():
            category = {'name': k, 'id': v}
            self.categories.append(category)

    def read_jsonfile(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def _image(self, obj, path):
        """
        构建coco的image字段
        :param obj:
        :param path:
        :return:
        """
        img_x = utils.img_b64_to_arr(obj['imageData'])
        h, w = img_x.shape[:-1]
        return {'height': h,
                'width': w,
                'id': self.image_id,
                'file_name': os.path.basename(path).replace(".json", ".jpg")}

    def _annotation(self, shape):
        """
        构建coco的annotation字段
        :param shape:
        :return:
        """
        label = shape['label']
        points = shape['points']
        contour = np.array(points)
        x = contour[:, 0]
        y = contour[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return {'id': self.annotation_id,
                'image_id': self.image_id,
                'category_id': int(CLASS_NAME_TO_ID[label]),
                'segmentation': np.asarray(points).flatten().tolist(),
                'bbox': self._get_box(points),
                'iscrowd': 0,
                'area': area}

    def _get_box(self, points):
        min_x = min_y = np.inf  # 无穷大
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]


if __name__ == '__main__':
    labelme_path = './images'
    save_coco_path = './coco/annotations'
    images_train_path = './coco/images/train2017'
    images_val_path = './coco/images/val2017'
    if not os.path.exists(save_coco_path):
        os.makedirs(save_coco_path)
    if not os.path.exists(images_train_path):
        os.makedirs(images_train_path)
    if not os.path.exists(images_val_path):
        os.makedirs(images_val_path)
    json_path_list = glob.glob(f'{labelme_path}/*.json')
    # 数据集划分
    train_path, val_path = train_test_split(json_path_list, test_size=0.15)
    print('train_num:', len(train_path), 'val_num:', len(val_path))
    # 转换成coco格式
    l2c_train = Labelme2Coco()
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, f'{save_coco_path}/instances_train2017.json')
    for file in train_path:
        shutil.copy(file.replace('json', 'jpg'), images_train_path)
    # 验证集
    l2c_val = Labelme2Coco()
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, f'{save_coco_path}/instances_val2017.json')
    for file in val_path:
        shutil.copy(file.replace('json', 'jpg'), images_val_path)
