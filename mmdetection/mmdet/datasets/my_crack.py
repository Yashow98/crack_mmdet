# -*- coding:utf-8 -*-
"""
@author:胡亚雄
@file: my_crack.py
@time: 2022/05/01
@description:
"""
from .coco import CocoDataset
from .builder import DATASETS


@DATASETS.register_module()
class MyCrack(CocoDataset):
    CLASSES = ('alligator', 'longitude', 'transverse', 'block', 'other')
