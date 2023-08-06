"""
Extracting the profile feature from input images of wikiperson dataset
"""

from transformers import BertTokenizer, BertModel
import json
import argparse
import logging
import os
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
import h5py
import clip
from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
Image.MAX_IMAGE_PIXELS = 2300000000  # 更改阈值像素上限


def wikiperson_input_profile(args):
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training
    model.eval()
    model.to(device)

    input_img_list = os.listdir(args.path_detection)
    img_id_list = []
    text_embeddings = torch.FloatTensor(len(input_img_list), 512)

    for index, item in tqdm(enumerate(input_img_list), total=len(input_img_list), desc="profile feature"):
        img_id = item.split(".")[0]
        img_id_list.append(img_id)
        path = os.path.join(args.path_detection, item)
        data = json.load(open(path, encoding="utf-8"))[0]

        gender = data['dominant_gender']
        race = data['dominant_race']
        age = data['age']
        emotion = data['dominant_emotion']

        profile = "gender: {}, race: {}, age: {}, emotion: {}".format(gender, race, age, emotion)
        profile_id = clip.tokenize(profile, truncate=True).to(device)
        with torch.no_grad():
            profile_feature = model.encode_text(profile_id).to(device)
        text_embeddings[index] = profile_feature

    json.dump(img_id_list, open(os.path.join(args.dir_output, "input_img_list.json"), 'w'), indent=2)
    text_file = h5py.File(os.path.join(args.dir_output, "profile.h5"), 'w')
    text_file.create_dataset("features", data=text_embeddings.numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_detection", default="../../data/wikiperson/detection", type=str)
    parser.add_argument("--path_entity", default="../../data/wikiperson/entity_list.json", type=str)
    parser.add_argument("--dir_output", default="../../data/wikiperson", type=str)
    args = parser.parse_args()

    wikiperson_input_profile(args)


if __name__ == "__main__":
    main()
