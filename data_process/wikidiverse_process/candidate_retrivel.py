import os
from utils import svg2png, gif2png
from tqdm import tqdm
import json
import random


wikidiverse_path = "E:\HIRWorks\data\ImgData\wikidiverse"

train_dataset_path = "E:\HIRWorks\data\wikidiverse\\train.json"
dev_dataset_path = "E:\HIRWorks\data\wikidiverse\\dev.json"
test_dataset_path = "E:\HIRWorks\data\wikidiverse\\test.json"

type2entity_path = "E:\HIRWorks\data\wikidiverse\\type2entity.json"

train = json.load(open(train_dataset_path, "r",encoding='utf-8'))
dev = json.load(open(dev_dataset_path, "r",encoding='utf-8'))
test = json.load(open(test_dataset_path, "r",encoding='utf-8'))

all_data = train + dev + test

type2entity = json.load(open(type2entity_path, "r"))

num_candidate = 100
all_candidate = {}
for i, item in tqdm(enumerate(all_data), total=len(all_data), ncols=80):
    mention_type = item["mention_type"]
    entity = item["entity"]
    candidate_set = set()
    # candidate_set.add(entity)
    for c in item["cands"]:
        candidate_set.add(c[0])
    extra_candidate = type2entity[mention_type]
    if len(extra_candidate) <= num_candidate - len(candidate_set):
        candidate_set = set.union(candidate_set, set(extra_candidate))
    else:
        ec_index_list = random.sample(range(0, len(extra_candidate) - 1), num_candidate-len(candidate_set))
        for ec_i in ec_index_list:
            candidate_set.add(extra_candidate[ec_i])

    all_candidate[entity] = list(candidate_set)

candidate_list_path = "E:\HIRWorks\data\wikidiverse\\search_top{}.json".format(num_candidate)
candidate_list_path = open(candidate_list_path, "w", encoding='utf-8')
candidate_list_path.write(json.dumps(all_candidate, indent=2))
