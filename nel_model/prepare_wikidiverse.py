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
    def __init__(self, guk, sent, mention, mention_type, image, answer):
        self.guk = guk  # The unique id of each example, generally composed of mode-key
        self.sent = sent
        self.mention = mention
        self.mention_type = mention_type
        self.image = image
        self.answer = answer


class InputFeatures:
    def __init__(self, guk, text_feature, mention_feature, total_feature, answer):
        self.guk = guk,
        self.text_feature = text_feature,
        self.mention_feature = mention_feature,
        self.total_feature = total_feature,
        self.answer = answer


class Wikidiverse():
    def __init__(self):
        super(Wikidiverse, self).__init__()
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
            examples.append(InputExample(
                guk=index,
                sent=item['sentence'],
                mention=item['mention'],
                mention_type=item['mention_type'],
                image=item['mention_image'],
                answer=item['entity'],
                )
            )

        return examples

    def convert_examples_to_features(self, examples):
        features = []
        for (ex_index, example) in tqdm(enumerate(examples), total=len(examples), ncols=80, desc="diverse:"):
            try:
                input_sent = example.mention + " [SEP] " + example.sent
                sent_ids = clip_tokenize(input_sent, truncate=True).to(self.device)  # 截断过长的
                mention = clip_tokenize(example.mention, truncate=True).to(self.device)
                answer = example.answer

                img_path = os.path.join(self.img_path, "wikidiverse", example.image)
                image = Image.open(img_path)
                image = self.preprocess(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    text_feature = self.model.encode_text(sent_ids)  # text_features 1,512
                    mention_feature = self.model.encode_text(mention)
                    total_feature = self.model.encode_image(image).to(self.device)

                features.append(
                    InputFeatures(
                        guk = example.guk,
                        text_feature = text_feature,
                        mention_feature = mention_feature,
                        total_feature=total_feature,
                        answer=answer
                    ))
            except Exception as e:
                print(e, example.image)
        return features
