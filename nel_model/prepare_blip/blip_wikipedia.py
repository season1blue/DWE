import logging
import os
import json
import torch
from PIL import Image
from tqdm import tqdm

from pixellib.torchbackend.instance import instanceSegmentation
from lavis.models import load_model_and_preprocess

import warnings
from args import parse_arg


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guk, sent, idx, answer=None, mentions=None, img_path=None):
        """Constructs a InputExample.
        Args:
        """
        self.guk = guk  # The unique id of each example, generally composed of mode-key
        self.sent = sent  # Sample text information
        self.img_id = idx  # The original id of the sample, used to retrieve the image
        self.answer = answer  # The answer information corresponding to the sample, that is, the id of the database instance
        self.mentions = mentions  # Reference information in the sample
        self.img_path = img_path


class InputFeatures:
    def __init__(self, answer_id, img_id, mentions, key_id, text_feature, mention_feature, total_feature,
                 segement_feature, profile_feature):
        self.answer_id = answer_id
        self.img_id = img_id
        self.mentions = mentions
        self.key_id = key_id
        self.text_feature = text_feature
        self.mention_feature = mention_feature
        self.total_feature = total_feature
        self.segement_feature = segement_feature
        self.profile_feature = profile_feature


class Wikipedia_blip():
    def __init__(self):
        super(Wikipedia_blip, self).__init__()
        args = parse_arg()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

        model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
                                                                          model_type="pretrain", is_eval=True,
                                                                          device=self.device)
        self.model = model
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors

        self.img_path = args.img_path
        Image.MAX_IMAGE_PIXELS = 2300000000  # 更改阈值像素上限


        # segment
        self.ins = instanceSegmentation()
        self.ins.load_model(args.seg_model_path)
        self.target_classes = self.ins.select_target_classes(person=True)
        self.detection_path = os.path.join(args.dir_prepro, "wiki_detection")
        self.segement_path = os.path.join(args.dir_prepro, "wiki_segement")
        self.total2part_map = json.load(open(os.path.join(args.dir_prepro, "total2part_map.json"), 'r'))

    def read_examples_from_file(self, data_dir, mode):
        file_path = os.path.join(data_dir, "{}.json".format(mode))
        examples = []

        js = json.load(open(file_path, encoding="utf-8"))

        for k, v in js.items():
            examples.append(
                InputExample(
                    guk=k,  # f"{mode}-{k}",
                    sent=v["sentence"],
                    idx=k,  # v["id"]
                    answer=v["answer"],  # v["answer"]
                    mentions=v["mentions"],
                    img_path=v['imgPath']
                )
            )

        return examples

    def convert_examples_to_features(self, examples):
        features = []
        for (ex_index, example) in tqdm(enumerate(examples), total=len(examples), ncols=80):
            self.model.to(self.device)
            img_id = example.img_path.split("/")[-1].split(".")[0]
            with torch.no_grad():
                input_sent = example.mentions + " [SEP] " + example.sent
                sent_ids = self.txt_processors["eval"](input_sent)
                mention = self.txt_processors["eval"](example.mentions)

                text_feature = self.model.extract_features({"image": None, "text_input": [sent_ids]}, mode="text").text_embeds[:, 0, :]
                mention_feature = self.model.extract_features({"image": None, "text_input": [mention]}, mode="text").text_embeds[:, 0, :]

                # extract image feature (split or single)
                img_path = os.path.join(self.img_path, example.img_path)

                total_image = Image.open(img_path).convert("RGB")
                total_image = self.vis_processors["eval"](total_image).unsqueeze(0).to(self.device)
                total_feature = self.model.extract_features({"image": total_image, "text_input": None}, mode="image").image_embeds[:, 0, :]

                if img_id not in self.total2part_map:
                    segement_features = torch.zeros_like(text_feature)
                    profile_features = torch.zeros_like(text_feature)
                else:
                    segement_list, profile_list = [], []
                    for part in self.total2part_map[img_id]:
                        # segement image feature extraction
                        segement_path = os.path.join(self.segement_path, part + ".jpg")
                        segement = Image.open(segement_path).convert("RGB")
                        segement = self.vis_processors["eval"](segement).unsqueeze(0).to(self.device)
                        segement_feature = self.model.extract_features({"image": segement, "text_input": None}, mode="image").image_embeds[:, 0, :]
                        segement_list.append(segement_feature)

                        # detection profile feature extraction
                        detection_path = os.path.join(self.detection_path, part + ".json")
                        detection_context = json.load(open(detection_path, 'r'))[0]
                        gender, race, age, emotion = detection_context['dominant_gender'], detection_context[
                            'dominant_race'], detection_context['age'], detection_context['dominant_emotion']
                        profile = "gender: {}, race: {}, age: {}, emotion: {}".format(gender, race, age, emotion)

                        profile = self.txt_processors["eval"](profile)
                        profile_feature = self.model.extract_features({"image": None, "text_input": [profile]}, mode="text").text_embeds[:, 0, :]
                        profile_list.append(profile_feature)

                    segement_features = torch.cat(segement_list, dim=0)
                    profile_features = torch.cat(profile_list, dim=0)

            answer_id = example.answer if example.answer else -1

            features.append(
                InputFeatures(
                    answer_id=answer_id,
                    img_id=example.img_id,
                    mentions=example.mentions,
                    key_id=example.guk,
                    text_feature=text_feature,
                    mention_feature=mention_feature,
                    total_feature=total_feature,
                    segement_feature=segement_features,
                    profile_feature=profile_features,
                )
            )
        return features
