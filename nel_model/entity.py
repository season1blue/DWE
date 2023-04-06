import logging
import os
import json
import torch
from PIL import Image
import clip
from tqdm import tqdm

from mlip.clip import load as clip_load
from mlip.clip import tokenize as clip_tokenize
from pixellib.torchbackend.instance import instanceSegmentation

import warnings
from args import parse_arg


class EntityExample(object):
    def __init__(self, id, sent, image=None):
        self.id = id  # The unique id of each example, generally composed of mode-key
        self.sent = sent  # Sample text information
        self.image = None


class EntityFeatures:
    def __init__(self, id, text_feature, image_feature=None):
        self.id = id
        self.text_feature = text_feature
        self.image_feature = image_feature


class Entity():
    def __init__(self):
        super(Entity, self).__init__()
        args = parse_arg()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
        model_path = os.path.join(args.pretrain_model_path, "ViT-B-32.pt")
        model, preprocess = clip_load(model_path, device=self.device, jit=False)
        self.model = model
        self.preprocess = preprocess

        self.img_path = args.img_path
        Image.MAX_IMAGE_PIXELS = 2300000000  # 更改阈值像素上限

    def read_entities_from_file(self, data_dir, filename):
        file_path = os.path.join(data_dir, filename)
        entities = []

        data = json.load(open(file_path, encoding="utf-8"))

        keys_ordered = sorted(list(data.keys()))
        for key in keys_ordered:
            entities.append(EntityExample(id=key, sent=data[key], image=None))

        return entities

    def convert_examples_to_features(self, examples):
        features = []
        for (ex_index, example) in tqdm(enumerate(examples), total=len(examples), ncols=80, desc="Entity:"):
            id = example.id
            sent = example.sent
            sent = sent.split("[SEP]. ")[-1]
            sent_ids = clip.tokenize(sent, truncate=True).to(self.device)
            with torch.no_grad():
                self.model.to(self.device)
                text_feature = self.model.encode_text(sent_ids)

            features.append(
                EntityFeatures(
                    id=id,
                    text_feature=text_feature,
                    image_feature=None
                ))
        return features
