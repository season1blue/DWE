import logging
import os
import json
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import clip
from tqdm import tqdm

from mlip.clip import load as clip_load
from mlip.clip import tokenize as clip_tokenize
from pixellib.torchbackend.instance import instanceSegmentation

import warnings
from args import parse_arg

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class InputExample(object):
    def __init__(self, guk, sent, idx, answer=None, mentions=None, img_list=None):
        self.guk = guk  # The unique id of each example, generally composed of mode-key
        self.sent = sent  # Sample text information
        self.img_id = idx  # The original id of the sample, used to retrieve the image
        self.answer = answer  # The answer information corresponding to the sample, that is, the id of the database instance
        self.mentions = mentions  # Reference information in the sample
        self.img_list = img_list


class InputFeatures:
    """A single training/test example for token classification."""

    def __init__(self, answer_id, img_id, mentions, key_id, text_feature, image_feature, mention_feature,
                 total_image_feature):
        self.answer_id = answer_id
        self.img_id = img_id
        self.mentions = mentions
        self.key_id = key_id
        self.text_feature = text_feature
        self.image_feature = image_feature
        self.mention_feature = mention_feature
        self.total_image_feature = total_image_feature


class Richpedia():
    def __init__(self):
        super(Richpedia, self).__init__()
        args = parse_arg()
        self.args = args

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_path = os.path.join(args.pretrain_model_path, "ViT-B-32.pt")
        model, preprocess = clip_load(model_path, device=self.device, jit=False)
        self.model = model
        self.preprocess = preprocess

        self.img_path = args.img_path
        Image.MAX_IMAGE_PIXELS = 2300000000  # 更改阈值像素上限

        self.cache_path = args.cache_path

        # segment
        self.ins = instanceSegmentation()
        self.ins.load_model(args.seg_model_path)
        self.target_classes = self.ins.select_target_classes(person=True)

    def read_examples_from_file(self, data_dir, mode):
        file_path = os.path.join(data_dir, "{}.json".format(mode))
        examples = []

        js = json.load(open(file_path, encoding="utf-8"))

        for k, v in js.items():
            # 过滤非jpg格式的图片
            img_list = []
            # 如果没有"img_list"字段，则返回空的img_list，共83个空
            if "img_list" in v.keys():
                for il in v['img_list']:
                    if il.split(".")[-1].lower() == "jpg":
                        img_list.append(il)

            examples.append(
                InputExample(
                    guk=k,  # f"{mode}-{k}",
                    sent=v["sentence"],
                    idx=k,  # v["id"]
                    answer=v["answer"],  # v["answer"]
                    mentions=v["mentions"],
                    img_list=img_list
                )
            )
        return examples

    # segement image to person image
    def split_image(self, img_path):
        # make sure delete the past segement result
        if len(os.listdir(self.cache_path)) != 0:
            for file in os.listdir(self.cache_path):
                os.remove(os.path.join(self.cache_path, file))

        self.ins.segmentImage(img_path, show_bboxes=True, extract_segmented_objects=True,
                              segment_target_classes=self.target_classes, save_extracted_objects=True,
                              output_path=self.cache_path)

        image_list = []
        for file in os.listdir(self.cache_path):
            file_path = os.path.join(self.cache_path, file)
            image = Image.open(file_path)
            image = self.preprocess(image)
            image = image.unsqueeze(0).to(self.device)
            image_list.append(image)
        return image_list

    def convert_examples_to_features(self, examples):
        features = []
        for (ex_index, example) in tqdm(enumerate(examples), total=len(examples), ncols=80):

            # Text
            input_sent = example.mentions + " [SEP] " + example.sent
            sent_ids, _ = clip_tokenize(input_sent, truncate=True)  # 截断过长的
            sent_ids = sent_ids.to(self.device)
            mention, _ = clip_tokenize(example.mentions, truncate=True)
            mention = mention.to(self.device)
            with torch.no_grad():
                self.model.to(self.device)
                text_feature = self.model.encode_text(sent_ids).to(self.device)  # text_features 1,512
                mention_feature = self.model.encode_text(mention).to(self.device)

            # Image
            image_features = []
            with torch.no_grad():
                for img_name in example.img_list:
                    img_path = os.path.join(self.img_path, img_name)
                    image = Image.open(img_path)
                    image = self.preprocess(image)
                    image = image.unsqueeze(0).to(self.device)
                    split_feature = self.model.encode_image(image).to(self.device)
                    image_features.append(split_feature)

                if len(image_features) == 0:
                    image_features = torch.zeros(1, 512).to(self.device)
                else:
                    image_features = torch.cat(image_features, dim=0).to(self.device)

            total_image_feature = torch.sum(image_features, dim=0).unsqueeze(0).to(self.device)  # 1, 512


            if example.answer:
                answer_id = example.answer
            else:
                answer_id = -1

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guk)
                logger.info("answer_id: %s", answer_id)

            features.append(
                InputFeatures(
                    answer_id=answer_id,
                    img_id=example.img_id,
                    mentions=example.mentions,
                    key_id=example.guk,
                    text_feature=text_feature,
                    image_feature=image_features,
                    mention_feature=mention_feature,
                    total_image_feature=total_image_feature
                )
            )
        return features
