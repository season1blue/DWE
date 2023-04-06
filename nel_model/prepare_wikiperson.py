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


class InputExample(object):
    def __init__(self, guk, image, answer, bounding_box):
        self.guk = guk  # The unique id of each example, generally composed of mode-key
        self.image = image
        self.answer = answer
        self.bounding_box = bounding_box


class InputFeatures:
    def __init__(self, img_id, image_feature, answer):
        self.img_id = img_id
        self.image_feature = image_feature
        self.answer = answer


class Wikiperson():
    def __init__(self):
        super(Wikiperson, self).__init__()
        self.args = parse_arg()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
        model_path = os.path.join(self.args.pretrain_model_path, "ViT-B-32.pt")
        model, preprocess = clip_load(model_path, device=self.device, jit=False)
        self.model = model
        self.preprocess = preprocess

        self.img_path = self.args.img_path
        Image.MAX_IMAGE_PIXELS = 2300000000  # 更改阈值像素上限

    def read_examples_from_file(self, data_dir, mode):
        file_path = os.path.join(data_dir, "{}.json".format(mode))
        examples = []

        data = json.load(open(file_path, encoding="utf-8"))

        for index, item in enumerate(data):
            eid = item["id"]
            image = item["image"]
            answer = item["wikiid"]
            bounding_box = item["boundingbox"]

            examples.append(InputExample(guk=eid, image=image, answer=answer, bounding_box=bounding_box))

        return examples

    def convert_examples_to_features(self, examples):
        features = []
        for (ex_index, example) in tqdm(enumerate(examples), total=len(examples), ncols=80, desc="person:"):
            img_id = example.image.split(".jpg")[0]
            answer = example.answer

            img_path = os.path.join(self.img_path, self.args.dataset, example.image)

            image = Image.open(img_path)
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_feature = self.model.encode_image(image).to(self.device)

            features.append(
                InputFeatures(
                    img_id=img_id,
                    image_feature=image_feature,
                    answer=answer
                ))
        return features
