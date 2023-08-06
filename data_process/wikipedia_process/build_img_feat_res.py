import torch.utils.data as data
import torch
import numpy as np
import json
import os
import h5py
import numpy
import torch
from PIL import Image
import requests

from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from PIL import Image
from tqdm import tqdm

from PIL import ImageFile
from build_text import load_data

from transformers import AutoFeatureExtractor, ResNetModel
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

data_path = "../data/new_data"
wiki_path = "../data/outside_data"

device = torch.device("cuda:0")
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-152")
model = ResNetModel.from_pretrained("microsoft/resnet-152") # 152 层数 follow clip flip
model.to(device)

import matplotlib.pyplot as plt

def build_img_list():
    dataset = load_data()
    id2imgpath = {}
    for id, data in dataset.items():
        id2imgpath[id] = data['imgPath']

    json.dump(id2imgpath, open(os.path.join(data_path, "img_list.json"), mode='w', encoding='utf8'), indent=1)
    return id2imgpath


class Cifar100(data.Dataset):
    # dirname 为训练/测试数据地址，使得训练/测试分开
    def __init__(self, img_list):
        super(Cifar100, self).__init__()
        # self.img_list = json.load(open(os.path.join(data_path, "img_list.json"))).values()
        self.path_list = list(img_list.values())
        self.name_list = list(img_list.keys())

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        with torch.no_grad():
            img_name = self.name_list[index]
            img_path = os.path.join(wiki_path, self.path_list[index])
            image = Image.open(img_path).convert("RGB")
            feature = feature_extractor(images=image, return_tensors="pt").to(device)
            outputs = model(**feature)
            hidden_states = outputs.last_hidden_state.to(device)
            hidden_states = hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)

        return hidden_states, img_name


def main():
    img_list = build_img_list()
    dataset = Cifar100(img_list)
    batch_size = 1
    dataloder = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=False)
    # img_feature = torch.FloatTensor(len(dataset), 197, 768)

    # ensure a new h5py file
    output_path = os.path.join(data_path, "img_feats_res.h5")
    if os.path.exists(output_path):
        os.remove(output_path)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloder, desc="data loading")):
            last_hidden_states, img_name = batch
            last_hidden_states = last_hidden_states[0]
            img_name = img_name[0]
            h5_file = h5py.File(os.path.join(data_path, "img_feats_res.h5"), 'a')
            h5_file.create_dataset(img_name, data=last_hidden_states.cpu().numpy())  #img_name 10-1

    # img_feature = torch.cat(feature_list, dim=0)
    # h5_file = h5py.File(os.path.join(data_path, "img_feats.h5"), 'w')
    # print("creating feature....")
    # h5_file.create_dataset("features", data=img_feature)

    # img_feat_h5 = h5py.File(os.path.join(data_path, "img_feats.h5"), 'r')
    # res = torch.from_numpy(img_feat_h5.get("10-1")[0])
    # print("res", res)

if __name__ == "__main__":
    main()
