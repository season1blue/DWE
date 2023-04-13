"""
    -----------------------------------
    dataset of nel
"""
import time

import torch
from torch.utils.data import Dataset
import h5py
import json
import re
import random
import numpy as np
import os

import torch
import numpy as np
import json
import h5py

from prepare_wikipedia import Wikipedia
from prepare_richpedia import Richpedia
from prepare_wikiperson import Wikiperson
from prepare_wikidiverse import Wikidiverse
from entity import Entity
from os.path import join, exists

INAME_PATTERN = re.compile("(\d+)\.")


def train_collate_fn(batch):
    answer_id_list, mention_feature_list, text_feature_list, segement_feature_list, total_feature_list, profile_feature_list, pos_sample_list, neg_sample_list = [], [], [], [], [], [], [], []

    for b in batch:
        answer_id, mention_feature, text_feature, segement_feature, total_feature, profile_feature, pos_sample, neg_sample = b.values()
        # print(neg_sample.size())
        # exit()
        mention_feature_list.append(mention_feature)
        text_feature_list.append(text_feature)
        segement_feature_list.append(segement_feature)
        total_feature_list.append(total_feature)
        profile_feature_list.append(profile_feature)
        pos_sample_list.append(pos_sample)
        neg_sample_list.append(neg_sample)

    # TODO: in-batch negatives
    # for i, b in enumerate(batch):
    #     neg = None
    #     answer_id, mention_feature, text_feature, segement_feature, total_feature, pos_sample, neg_sample = b.values()
    #     while neg is None:
    #         rand = random.randint(0, len(pos_sample_list) - 1)  # randint [0,x] 闭区间
    #         if rand != i:
    #             neg = pos_sample_list[rand]
    #     neg_sample_list[i] = torch.cat([neg_sample_list[i], neg], dim=0)

    max_size = max([imf.size(0) for imf in segement_feature_list])  # img_feature.size == (n, 512)

    mention_feature = torch.stack(mention_feature_list)
    total_feature = torch.stack(total_feature_list)
    text_feature = torch.stack(text_feature_list)
    pos_sample = torch.stack(pos_sample_list)
    neg_sample = torch.stack(neg_sample_list)

    for imf_index in range(len(segement_feature_list)):
        while segement_feature_list[imf_index].size(0) < max_size:
            segement_feature_list[imf_index] = torch.nn.functional.pad(segement_feature_list[imf_index],
                                                                       pad=(0, 0, 0, 1),
                                                                       mode='constant', value=0)
            profile_feature_list[imf_index] = torch.nn.functional.pad(profile_feature_list[imf_index], pad=(0, 0, 0, 1),
                                                                      mode='constant', value=0)

    segement_feature = torch.stack(segement_feature_list)
    profile_feature = torch.stack(profile_feature_list)

    return {
        "mention_feature": mention_feature,
        "text_feature": text_feature,
        "total_feature": total_feature,
        "segement_feature": segement_feature,
        "profile_feature": profile_feature,
        "pos": pos_sample,
        "neg": neg_sample,
    }


def eval_collate_fn(batch):
    answer_id_list, mention_feature_list, text_feature_list, segement_feature_list, total_feature_list, profile_feature_list, pos_sample_list, neg_sample_list, search_res_list = [], [], [], [], [], [], [], [], []

    for b in batch:
        answer_id, mention_feature, text_feature, segement_feature, total_feature, profile_feature, pos_sample, neg_sample, search_res = b.values()
        # print(neg_sample.size())
        # exit()
        mention_feature_list.append(mention_feature)
        text_feature_list.append(text_feature)
        segement_feature_list.append(segement_feature)
        total_feature_list.append(total_feature)
        profile_feature_list.append(profile_feature)
        pos_sample_list.append(pos_sample)
        neg_sample_list.append(neg_sample)
        search_res_list.append(search_res)

    # TODO: in-batch negatives
    # for i, b in enumerate(batch):
    #     neg = None
    #     answer_id, mention_feature, text_feature, segement_feature, total_feature, pos_sample, neg_sample = b.values()
    #     while neg is None:
    #         rand = random.randint(0, len(pos_sample_list) - 1)  # randint [0,x] 闭区间
    #         if rand != i:
    #             neg = pos_sample_list[rand]
    #     neg_sample_list[i] = torch.cat([neg_sample_list[i], neg], dim=0)

    max_size = max([imf.size(0) for imf in segement_feature_list])  # img_feature.size == (n, 512)

    mention_feature = torch.stack(mention_feature_list)
    total_feature = torch.stack(total_feature_list)
    text_feature = torch.stack(text_feature_list)
    pos_sample = torch.stack(pos_sample_list)
    neg_sample = torch.stack(neg_sample_list)
    search_res = torch.stack(search_res_list)

    for imf_index in range(len(segement_feature_list)):
        while segement_feature_list[imf_index].size(0) < max_size:
            segement_feature_list[imf_index] = torch.nn.functional.pad(segement_feature_list[imf_index],
                                                                       pad=(0, 0, 0, 1),
                                                                       mode='constant', value=0)
            profile_feature_list[imf_index] = torch.nn.functional.pad(profile_feature_list[imf_index], pad=(0, 0, 0, 1),
                                                                      mode='constant', value=0)

    segement_feature = torch.stack(segement_feature_list)
    profile_feature = torch.stack(profile_feature_list)

    return {
        "mention_feature": mention_feature,
        "text_feature": text_feature,
        "total_feature": total_feature,
        "segement_feature": segement_feature,
        "profile_feature": profile_feature,
        "pos": pos_sample,
        "neg": neg_sample,
        "search_res": search_res,
    }


def neg_sample_online(neg_id, neg_iid, tfidf_neg, negid2qid, max_sample_num=1, threshold=0.95):
    """
        Online negative sampling algorithm
        ------------------------------------------
        Args:
        Returns:
    """
    N = len(tfidf_neg)
    cands = set()

    while len(cands) < max_sample_num:
        rand = random.random()
        # print("neg id", neg_id, tfidf_neg[neg_id])
        if not tfidf_neg[neg_id] or rand > threshold:
            cand = random.randint(0, N - 1)
        else:
            rand_word = random.choice(tfidf_neg[neg_id])
            cand = random.choice(neg_iid[rand_word])
            # print("tfidf", tfidf_neg[neg_id])
            # print("randword", rand_word)
            # print("negidd", neg_iid[rand_word])

        if cand != neg_id:
            cands.add(cand)

    return [negid2qid[c] for c in cands]


def neg_sample(entity_list, pos_id, max_sample_num):
    candidate = set()
    while len(candidate) < max_sample_num:
        rand = random.randint(0, len(entity_list) - 1)  # randint [0,x] 闭区间
        if rand != pos_id:
            candidate.add(rand)
    return list(candidate)


def search_res(entity_list, pos_id, max_sample_num=1000):
    candidate = random.sample(range(0, len(entity_list) - 1), max_sample_num)
    candidate = [pos_id] + candidate
    return candidate


class NELDataset(Dataset):
    def __init__(self, args,
                 all_answer_id,
                 all_mentions,
                 all_img_id,
                 all_key_id,
                 all_mention_feature,
                 all_text_feature,
                 all_total_feature,
                 all_segement_feature,
                 all_profile_feature,
                 answer_list,
                 contain_search_res=False):
        # text info
        self.all_answer_id = all_answer_id
        self.all_img_id = all_img_id
        self.all_mentions = all_mentions
        self.all_mention_feature = all_mention_feature
        # answer
        self.answer_list = answer_list  # id2ansStr #qids_ordered.json
        self.answer_mapping = {answer: i for i, answer in enumerate(self.answer_list)}  # ansStr2id

        # Online negative sampling
        self.max_sample_num = args.neg_sample_num
        neg_config = json.load(open(args.path_neg_config))
        self.neg_iid = neg_config["neg_iid"]
        self.tfidf_neg = neg_config["tfidf_neg"]
        self.negid2qid = neg_config["keys_ordered"]
        self.qid2negid = {qid: i for i, qid in enumerate(neg_config["keys_ordered"])}

        # Sample features of negative sampling
        self.neg_list = json.load(open(join(args.dir_neg_feat, "qids_ordered.json")))  # len = 25846
        self.neg_mapping = {sample: i for i, sample in enumerate(self.neg_list)}
        self.ansid2negid = {i: self.neg_mapping[ans] for i, ans in enumerate(self.answer_list)}

        gt_name = "gt_feats_{}.h5".format(args.gt_type)
        entity_feat = h5py.File(join(args.dir_neg_feat, gt_name), 'r')
        self.entity_features = entity_feat.get("features")

        # search candidates
        self.contain_search_res = contain_search_res
        if self.contain_search_res:
            self.search_res = json.load(
                open(args.path_candidates, "r", encoding='utf8'))  # mention: [qid0, qid1, ..., qidn]

        self.all_text_features = all_text_feature
        self.all_total_features = all_total_feature
        self.all_segement_features = all_segement_feature
        self.all_profile_features = all_profile_feature

    def __len__(self):
        return len(self.all_answer_id)

    def __getitem__(self, idx):
        sample = dict()
        sample["answer_id"] = self.all_answer_id[idx]
        sample["mention_feature"] = self.all_mention_feature[idx]
        sample["text_feature"] = self.all_text_features[idx]
        sample['segement_feature'] = self.all_segement_features[idx]
        sample['total_feature'] = self.all_total_features[idx]
        sample['profile_feature'] = self.all_profile_features[idx]

        ans_id = self.all_answer_id[idx]
        if ans_id == "c":
            ans_id = "Q5729149"
        pos_sample_id = self.neg_mapping[ans_id]
        neg_ids = neg_sample_online(self.qid2negid[ans_id], self.neg_iid, self.tfidf_neg, self.negid2qid,
                                    self.max_sample_num)
        neg_ids_map = [self.neg_mapping[nid] for nid in neg_ids]

        sample["pos_sample"] = torch.tensor(np.array([self.entity_features[pos_sample_id]]))
        sample["neg_sample"] = torch.tensor(np.array([self.entity_features[nim] for nim in neg_ids_map]))

        # return search results
        if self.contain_search_res:
            qids_searched = self.search_res[self.all_mentions[idx]]
            qids_searched_map = [self.neg_mapping[qid] for qid in qids_searched]
            sample["search_res"] = torch.tensor(np.array([self.entity_features[qsm] for qsm in qids_searched_map]))
            # print(sample["search_res"].size())  #Bathc_size*hidden_size 32*768
        return sample


def person_collate_train(batch):
    image_feature_list, detection_list, pos_list, neg_list = [], [], [], []
    answer_list = []

    for b in batch:
        answer, image_feature, detection, pos = b.values()
        answer_list.append(answer)
        image_feature_list.append(image_feature)
        detection_list.append(detection)
        pos_list.append(pos)

    for i, b in enumerate(batch):
        neg = None
        _, _, _, pos = b.values()
        while neg is None:
            rand = random.randint(0, len(pos_list) - 1)  # randint [0,x] 闭区间
            if rand != i:
                neg = pos_list[rand]
        neg_list.append(neg)

    answer = torch.tensor(answer_list)
    image_feature = torch.cat(image_feature_list, dim=0)  # [bs, hidden_size]
    detection = torch.cat(detection_list, dim=0)  # [bs, hidden_size]
    pos = torch.cat(pos_list, dim=0).unsqueeze(1)  # [bs, 1, hidden_size]
    neg = torch.cat(neg_list, dim=0)  # [bs, hidden_size]

    return {
        "answer": answer,
        "image_feature": image_feature,
        "detection": detection,
        "pos": pos,
        "neg": neg,
    }


def person_collate_eval(batch):
    image_feature_list, detection_list, pos_list, neg_list = [], [], [], []
    candidate_list = []
    answer_list = []

    for b in batch:
        answer, image_feature, detection, pos, candidate = b.values()
        answer_list.append(answer)
        image_feature_list.append(image_feature)
        detection_list.append(detection)
        pos_list.append(pos)
        candidate_list.append(candidate.unsqueeze(0))

    for i, b in enumerate(batch):
        neg = None
        _, _, _, pos, _ = b.values()
        while neg is None:
            rand = random.randint(0, len(pos_list) - 1)  # randint [0,x] 闭区间
            if rand != i:
                neg = pos_list[rand]
        neg_list.append(neg)

    answer = torch.tensor(answer_list)
    image_feature = torch.cat(image_feature_list, dim=0)  # [bs, hidden_size]
    detection = torch.cat(detection_list, dim=0)  # [bs, hidden_size]
    pos = torch.cat(pos_list, dim=0).unsqueeze(1)  # [bs, 1, hidden_size]
    neg = torch.cat(neg_list, dim=0)  # [bs, hidden_size]
    candidate = torch.cat(candidate_list, dim=0)

    # candidate_list = pos_list
    # random.shuffle(candidate_list)
    # candidate = torch.cat(candidate_list, dim=0).unsqueeze(1)
    # candidate = candidate.repeat(1, len(pos_list), 1)  # [bs, bs, hidden_size]

    return {
        "answer": answer,
        "image_feature": image_feature,
        "detection": detection,
        "pos": pos,
        "neg": neg,
        "search_res": candidate
    }


class DiverseDataset(Dataset):
    def __init__(self, args, guks, all_text_feature, all_mention_feature, all_total_feature, all_answer_id,
                 contain_search_res):
        self.args = args
        self.guks = guks
        self.all_answer_id = all_answer_id
        self.all_text_feature = all_text_feature
        self.all_mention_feature = all_mention_feature
        self.all_total_feature = all_total_feature
        self.max_sample_num = args.neg_sample_num

        self.entity_list = json.load(open(join(args.dir_neg_feat, "entity_list.json")))  # len = 25846
        self.entity_id_mapping = {sample: i for i, sample in enumerate(self.entity_list)}

        # profile_path = os.path.join(args.dir_neg_feat, "profile.h5")
        # self.profile = h5py.File(profile_path, 'r').get("features")

        entity_feat = h5py.File(join(args.dir_neg_feat, "gt_feats_brief.h5"), 'r')
        self.entity_features = entity_feat.get("features")

        self.contain_search_res = contain_search_res
        self.neg_sample_list = json.load(open(args.path_candidates, "r", encoding='utf8'))

    def __len__(self):
        return len(self.all_answer_id)

    def __getitem__(self, idx):
        sample = dict()

        print(self.guks[idx])
        print(idx)
        ans_qid = self.all_answer_id[idx]  # Q3290309
        pos_sample_id = self.entity_id_mapping[ans_qid]  # 46729
        neg_sample_qids = self.neg_sample_list[ans_qid][:min(len(self.neg_sample_list[ans_qid]),
                                                             self.max_sample_num)]  # ['Q12586851', 'Q2929059', 'Q4720236', 'Q19958130'
        neg_sample_ids = [self.entity_id_mapping[qids] for qids in
                          neg_sample_qids]  # [46729, 25088, 32776, 116770, 96808.. ] 100

        sample["answer_id"] = self.entity_id_mapping[self.all_answer_id[idx]]
        sample['image_feature'] = self.all_total_feature[idx]
        sample['text_feature'] = self.all_text_feature[idx]
        sample['mention_feature'] = self.all_mention_feature[idx]

        sample["pos"] = torch.tensor(np.array([self.entity_features[pos_sample_id]]))
        sample["neg"] = torch.tensor(np.array([self.entity_features[nim] for nim in neg_sample_ids]))

        if self.contain_search_res:
            qids_searched = self.neg_sample_list[ans_qid]
            qids_searched_map = [pos_sample_id] + [self.entity_id_mapping[qid] for qid in qids_searched]
            sample["search_res"] = torch.tensor(np.array([self.entity_features[qsm] for qsm in qids_searched_map]))
        return sample


class PersonDataset(Dataset):
    def __init__(self, args, all_img_id, all_answer_id, all_image_feature, contain_search_res):
        self.args = args
        self.all_img_id = all_img_id
        self.all_answer_id = all_answer_id
        self.all_image_features = all_image_feature
        self.max_sample_num = args.neg_sample_num

        self.entity_list = json.load(open(join(args.dir_neg_feat, "entity_list.json")))  # len = 25846
        self.entity_id_mapping = {sample: i for i, sample in enumerate(self.entity_list)}
        self.neg_config = json.load(open(args.path_neg_config))
        self.qid2negid = {qid: i for i, qid in enumerate(self.neg_config["keys_ordered"])}

        entity_list = json.load(open(self.args.path_ans_list))
        self.entity_mapping = {sample: i for i, sample in enumerate(entity_list)}

        img_list = json.load(open(join(args.dir_neg_feat, "input_img_list.json")))
        self.img_mapping = {sample: i for i, sample in enumerate(img_list)}

        profile_path = os.path.join(args.dir_neg_feat, "profile.h5")
        self.profile = h5py.File(profile_path, 'r').get("features")

        if args.gt_type != "both":
            gt_name = "{}_entity.h5".format(args.gt_type)
            entity_feat = h5py.File(join(args.dir_neg_feat, gt_name), 'r')
            self.entity_features = entity_feat.get("features")
        else:
            entity_image_feat = h5py.File(join(args.dir_neg_feat, "image_entity.h5"), 'r')
            entity_text_feat = h5py.File(join(args.dir_neg_feat, "text_entity.h5"), 'r')
            self.visual_entity_features = entity_image_feat.get("features")
            self.textual_entity_features = entity_text_feat.get("features")

        self.contain_search_res = contain_search_res
        self.neg_sample_list = json.load(open(args.path_candidates, "r", encoding='utf8'))

    def __len__(self):
        return len(self.all_answer_id)

    def __getitem__(self, idx):
        sample = dict()

        img_id = self.all_img_id[idx]
        ans_qid = self.all_answer_id[idx]  # Q3290309
        pos_sample_id = self.entity_id_mapping[ans_qid]  # 46729
        neg_sample_qids = self.neg_sample_list[ans_qid][:min(len(self.neg_sample_list[ans_qid]),
                                                             self.max_sample_num)]  # ['Q12586851', 'Q2929059', 'Q4720236', 'Q19958130'
        neg_sample_ids = [self.entity_id_mapping[qids] for qids in
                          neg_sample_qids]  # [46729, 25088, 32776, 116770, 96808.. ] 100

        sample["answer_id"] = self.entity_mapping[self.all_answer_id[idx]]
        sample['image_feature'] = self.all_image_features[idx]
        sample["detection"] = torch.tensor(np.array([self.profile[self.img_mapping[img_id]]]))

        if self.args.gt_type != "both":
            sample["pos"] = torch.tensor(np.array([self.entity_features[pos_sample_id]]))
            sample["neg"] = torch.tensor(np.array([self.entity_features[nim] for nim in neg_sample_ids]))
        else:
            pos_textual_feature = torch.tensor(np.array([self.textual_entity_features[pos_sample_id]]))
            pos_visual_feature = torch.tensor(np.array([self.visual_entity_features[pos_sample_id]]))
            neg_textual_feature = torch.tensor(np.array([self.textual_entity_features[nim] for nim in neg_sample_ids]))
            neg_visual_feature = torch.tensor(np.array([self.visual_entity_features[nim] for nim in neg_sample_ids]))

            sample["pos"] = pos_textual_feature + pos_visual_feature
            sample["neg"] = neg_textual_feature + neg_visual_feature

        if self.contain_search_res:
            qids_searched = self.neg_sample_list[ans_qid]
            qids_searched_map = [pos_sample_id] + [self.entity_id_mapping[qid] for qid in qids_searched]
            sample["search_res"] = torch.tensor(np.array([self.entity_features[qsm] for qsm in qids_searched_map]))
        return sample


def load_and_cache_examples(args, tokenizer, answer_list, mode, dataset="wiki", logger=None):
    if dataset == "wiki":
        data_processor = Wikipedia()
    elif dataset == "rich":
        data_processor = Richpedia()
    elif dataset == "person":
        data_processor = Wikiperson()
    elif dataset == "diverse":
        data_processor = Wikidiverse()
    else:
        print("Specify the dataset name: wiki, rich, person, diverse")
        exit()

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.dir_prepro, "cached_{}_{}".format(mode, args.dataset))
    # entity_features_file = os.path.join(args.dir_prepro, "entity_{}".format(args.dataset))

    guks = []

    if mode != 'test' and os.path.exists(cached_features_file) and not args.overwrite_cache:
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features %s at %s" % (cached_features_file, args.dir_prepro))

        examples = data_processor.read_examples_from_file(args.dir_prepro, mode)
        features = data_processor.convert_examples_to_features(examples)

        guks = [ex.guk for ex in examples]

        if args.local_rank in [-1, 0] and not args.do_cross:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # ----- split -----
    contain_search_res = False if mode == "train" else True

    if args.dataset == "person":
        all_img_id = [f.img_id for f in features]
        all_answer_id = [f.answer for f in features]
        all_image_feature = [f.image_feature for f in features]
        dataset = PersonDataset(args, all_img_id, all_answer_id, all_image_feature, contain_search_res)
    elif args.dataset == "diverse":
        all_text_feature = [f.text_feature for f in features]
        all_mention_feature = [f.mention_feature for f in features]
        all_total_feature = [f.total_feature for f in features]

        dataset = DiverseDataset(args, guks, all_text_feature, all_mention_feature, all_total_feature,
                                 contain_search_res)


    else:
        all_text_feature = [f.text_feature for f in features]
        all_mention_feature = [f.mention_feature for f in features]
        all_total_feature = [f.total_feature for f in features]
        all_segement_feature = [f.segement_feature for f in features]
        all_profile_feature = [f.profile_feature for f in features]

        all_answer_id = [f.answer_id for f in features]
        all_img_id = [f.img_id for f in features]
        all_key_id = [f.key_id for f in features]
        all_mentions = [f.mentions for f in features]

        dataset = NELDataset(args,
                             all_answer_id,
                             all_mentions,
                             all_img_id,
                             all_key_id,
                             all_mention_feature,
                             all_text_feature,
                             all_total_feature,
                             all_segement_feature,
                             all_profile_feature,
                             answer_list,
                             contain_search_res)
    return dataset, guks
