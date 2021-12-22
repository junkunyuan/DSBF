# coding=utf-8
from dataset.static.data_label_mapping import *

from dataset.transforms import dg_train_transformer,dg_test_transformer,image_test,image_train, digit_transforms

ob_tr_transf = image_train(256, 224)
ob_te_transf = image_test(256, 224)
office_31={
    "class_num":31,
    "dataset_name":"office-31",
    "dataset_root_dir":"/mnt/sda3/Data_Free/Office-31",
    "data_list_dir":"dataset/pthList/office-31",
    "domains":["amazon","dslr","webcam"],
    "label2num":office_31_mapping,
    "tr_te_scale":[0.9,1],
    "transforms": [ob_tr_transf,ob_te_transf],
}

office_home={
    "class_num":65,
    "dataset_name": "office-home",
    "dataset_root_dir":"/mnt/sda3/Data_Free/office-home",
    "data_list_dir":"dataset/pthList/office-home",
    "domains":["Art","Clipart","Product","RealWorld"],
    "label2num":office_home_mapping,
    "tr_te_scale":[0.9,1],
    "transforms": [image_train(),image_test()],
}

dac = {
    "class_num": 2,
    "dataset_name": "dog vs cat",
    "dataset_root_dir": "/mnt/sda3/Data_Free/dac",
    "data_list_dir": "dataset/pthList/dac",
    "domains": ["1TB","2TB"],
    "label2num": None,
    "tr_te_scale": [0.9, 1],
    "transforms": [ob_tr_transf,ob_te_transf],
}

face_gender = {
    "class_num": 2,
    "dataset_name": "face_gender",
    "dataset_root_dir": "/mnt/sda3/Data_Free/face",
    "data_list_dir": "dataset/pthList/face_gender",
    "domains": ["1EB_gender","2EB_gender"],
    "label2num": None,
    "tr_te_scale": [0.9, 1],
    "transforms": [ob_tr_transf,ob_te_transf],
}

pacs={
    "class_num":7,
    "dataset_name":"pacs",
    "dataset_root_dir":"F:\datas\Shiyanshi\Done\PACS",
    "data_list_dir":"dataset/pthList/pacs",
    "domains":["art_painting","cartoon","photo","sketch"],
    "label2num":pacs_mapping,
    "tr_te_scale":[0.9,0.1],
    "transforms":[dg_train_transformer(222), dg_test_transformer(222)]
}

office_8 = {
    "class_num": 8,
    "dataset_name": "office-8",
    "dataset_root_dir": "/mnt/sda3/Data_Free/Office-31",
    "data_list_dir": "dataset/pthList/office-8",
    "domains": ["amazon", "dslr", "webcam","caltech","RealWorld","Art","Clipart","Product"],
    "label2num": None,
    "tr_te_scale": [0.9, 1],
    "transforms": [image_train(), image_test()],
}


def ConfigFactory(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name in ["office-31","office","office_31","office31"]:
        return office_31
    elif dataset_name in ["office-home","office_home","officehome"]:
        return office_home
    elif dataset_name in ["pacs"]:
        return pacs
    elif dataset_name in ["dac"]:
        return dac
    elif dataset_name in ["face_gender"]:
        return face_gender
    return None
