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
    """A single training/test example for token classification."""

    def __init__(self, answer_id, img_id, mentions, key_id, text_feature, image_feature, mention_feature,
                 total_image_feature):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.answer_id = answer_id
        self.img_id = img_id
        self.mentions = mentions
        self.key_id = key_id
        self.text_feature = text_feature
        self.image_feature = image_feature
        self.mention_feature = mention_feature
        self.total_image_feature = total_image_feature


class Wikipedia():
    def __init__(self):
        super(Wikipedia, self).__init__()
        args = parse_arg()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
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
            img_path = os.path.join(self.img_path, example.img_path)
            input_sent = example.mentions + " [SEP] " + example.sent
            sent_ids, _ = clip_tokenize(input_sent, truncate=True)  # 截断过长的
            sent_ids = sent_ids.to(self.device)
            mention, _ = clip_tokenize(example.mentions, truncate=True)
            mention = mention.to(self.device)

            image_features = []
            with torch.no_grad():
                self.model.to(self.device)
                text_feature = self.model.encode_text(sent_ids)  # text_features 1,512
                mention_feature = self.model.encode_text(mention)

                # extract image feature (split or single)
                split_list = self.split_image(img_path)
                if len(split_list) > 0:
                    for split in split_list:
                        split_feature = self.model.encode_image(split).to(self.device)
                        image_features.append(split_feature)
                    image_features = torch.cat(image_features, dim=0)
                else:
                    split_single_image = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
                    image_features = self.model.encode_image(split_single_image)

                total_image = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
                total_image_feature = self.model.encode_image(total_image)

            if example.answer:
                answer_id = example.answer
            else:
                answer_id = -1

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
