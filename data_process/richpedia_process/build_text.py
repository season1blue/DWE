import os
import json
import argparse
import pickle
from os.path import join, exists
from fuzzywuzzy import process
from multiprocessing.pool import Pool
from tqdm import tqdm
from sample_util import build_dict, build_inverted_index, build_tfidf, negative_sample
import random
STOP_WORDS = {",", ".", "!", ":", "'", "\"", ";", "/", "\\", "Sex", "Name", "Nick", "Occupation", "Birth", "Languages",
              "Religion", "Alma", "mater", ""}


def run(m, ne_list, num_search):
    return process.extract(m, ne_list, limit=num_search)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_dataset", default="../../data/richpedia",
                        help="Path to file 'dataset.json'")
    parser.add_argument("--num_search", default="100")
    # dataset_path = "../data/new_data"
    return parser.parse_args()


args = parse_args()
dataset_path = args.path_dataset



def split_dataset(portion=0.8):
    dataset = json.load(open(os.path.join(dataset_path, "Richpedia-MEL.json"), encoding='utf-8'))
    test = os.path.join(dataset_path, "Richpedia-MEL.json")
    print(test)
    exit()
    # train_data, test_data = train_test_split(dataset, test_size=0.2, stratify=None)
    train_data, test_data = {}, {}
    for key, value in dataset.items():
        if random.random() < portion:
            train_data[key] = value
        else:
            test_data[key] = value
    json.dump(train_data, open(os.path.join(dataset_path, "train.json"), mode='w', encoding='utf8'), indent=2)
    json.dump(test_data, open(os.path.join(dataset_path, "test.json"), mode='w', encoding='utf8'), indent=2)


def load_data():
    print("1 loading data")
    dataset = json.load(open(os.path.join(dataset_path, "Richpedia-MEL.json"), encoding='utf-8'))
    sorted_dataset = {}
    for key in sorted(dataset):
        sorted_dataset[key] = dataset[key]

    return sorted_dataset


def fact_build(dataset):
    print("2 building fact")
    fact, ne2qid, qid2id = {}, {}, {}
    for _, data in dataset.items():
        if "brief" in data.keys():
            # print(data['brief'])
            qid = data['answer']
            entity = data['entities']
            brief = data['brief']
            fact[qid] = entity + " [SEP]. " + brief

    # tmp_set = set()
    # for id, data in sorted_dataset.items():
    #     if data['answer'] not in fact.keys():
    #         tmp_set.add(data['answer'])
    # print(tmp_set)
    # print(len(tmp_set))  # 229个entity没有对应的brief

    for _, data in dataset.items():
        qid = data['answer']
        if qid not in fact.keys():  # len 17391
            entity = data['entities']
            fact[qid] = entity + " [SEP]. " + "None"

    json.dump(fact, open(os.path.join(dataset_path, "fact_supply.json"), mode='w', encoding='utf8'), indent=2)

    for id, data in dataset.items():
        qid = data['answer']
        entity = data['entities']
        ne2qid[entity] = qid
        qid2id[qid] = id  # len 17391

    # sorted
    sorted_fact = {}
    for key in sorted(fact):
        sorted_fact[key] = fact[key]
    qid_list = list(sorted_fact.keys())

    sorted_qid2id = {}
    for key in sorted(qid2id):
        sorted_qid2id[key] = qid2id[key]

    json.dump(sorted_fact, open(os.path.join(dataset_path, "fact.json"), mode='w', encoding='utf8'), indent=2)
    json.dump(qid_list, open(os.path.join(dataset_path, "qids_ordered.json"), mode='w', encoding='utf8'), indent=2)
    json.dump(ne2qid, open(os.path.join(dataset_path, "ne2qid.json"), mode='w', encoding='utf8'), indent=2)
    json.dump(sorted_qid2id, open(os.path.join(dataset_path, "qid2id.json"), mode='w', encoding='utf8'), indent=2)

    return fact, ne2qid


def neg(dataset, fact):
    print("3 neging")
    path_dict = os.path.join(dataset_path, "tfidf_dict.pkl")
    path_neg = os.path.join(dataset_path, "neg.json")
    keys_ordered = sorted(list(fact.keys()))

    # construct dicts
    dictionary, toksList_ordered = build_dict([fact[k] for k in keys_ordered])
    nwords = len(dictionary)

    # Turn each sentence into the form of ID-freq
    corpus = [dictionary.doc2bow(toks_list) for toks_list in toksList_ordered]  # sent 2 ids

    _, neg_iid = build_inverted_index(nwords, corpus)
    _, tfidf_neg = build_tfidf(corpus)

    pickle.dump(dictionary, open(path_dict, "wb"))
    json.dump({
        "keys_ordered": keys_ordered,
        "neg_iid": neg_iid,
        "tfidf_neg": tfidf_neg,
        "corpus": corpus,
    }, open(path_neg, 'w'), indent=2)  # neg.json


def search_fuzz():
    print(f"4 Searching candidates with fuzzywuzzy (num: {args.num_search}).")
    dataset = load_data()

    ne2qid = json.load(open(os.path.join(dataset_path, "ne2qid.json"), encoding='utf-8'))
    path_search_tmp = join(dataset_path, f"search_tmp{args.num_search}.json")

    print("Pooling")
    if True or not exists(path_search_tmp):
        ne_list = list(ne2qid.keys())
        key_list = list(dataset.keys())
        mentions = [dataset[key]["mentions"] for key in key_list]

        pool = Pool(40)
        search_res = [pool.apply_async(run, (m, ne_list, args.num_search,)) for m in mentions]
        search_tmp = [sr.get() for sr in search_res]
        json.dump({
            'search_res': search_tmp,
            'key_ordered': key_list
        }, open(path_search_tmp, 'w', encoding='utf-8'), indent=2)
    else:
        tmp = json.load(open(path_search_tmp, encoding='utf-8'))
        key_list = tmp['key_ordered']
        search_tmp = tmp["search_res"]

    print("searching")
    N = len(key_list)
    men2qids = dict()
    for i in range(N):
        mentions = dataset[key_list[i]]["mentions"]
        qids = []
        for item in search_tmp[i]:
            ne = item[0]
            qids.append(ne2qid[ne])

        men2qids[mentions] = qids
    json.dump(men2qids, open(os.path.join(dataset_path, f"search_top{args.num_search}.json"), 'w'), indent=2)
    print("Finish search kg fuzz")


# 处理数据集本身
def fact_supply_and_pure():
    fact_supplyment = json.load(open(os.path.join(dataset_path, "fact_supply.json"), encoding='utf-8'))
    delete_qid = []
    count = 0
    for qid, fs in fact_supplyment.items():
        if fs.split(". ")[-1] == "None":
            delete_qid.append(qid)
            count += 1

    train_data = json.load(open(os.path.join(dataset_path, "train.json"), encoding='utf-8'))
    test_data = json.load(open(os.path.join(dataset_path, "test.json"), encoding='utf-8'))
    dev_data = json.load(open(os.path.join(dataset_path, "dev.json"), encoding='utf-8'))

    new_train_data, new_dev_data, new_test_data = {}, {}, {}
    for id, data in train_data.items():
        if data['answer'] not in delete_qid:
            new_train_data[id] = data

    for id, data in test_data.items():
        if data['answer'] not in delete_qid:
            new_test_data[id] = data

    for id, data in dev_data.items():
        if data['answer'] not in delete_qid:
            new_dev_data[id] = data

    json.dump(new_train_data, open(os.path.join(dataset_path, "train.json"), 'w', encoding='utf8'), indent=2)
    json.dump(new_test_data, open(os.path.join(dataset_path, "test.json"), 'w', encoding='utf8'), indent=2)
    json.dump(new_dev_data, open(os.path.join(dataset_path, "dev.json"), 'w', encoding='utf8'), indent=2)
    # all_data = json.load(open(os.path.join(dataset_path, "all_dataset.json"), encoding='utf-8'))


def main():
    args = parse_args()
    # 用以将RichPedia-MEL.json划分成train和test数据集
    split_dataset()

    # 处理缺失的brief（手动添加到fact_supply.json文件中的）
    # fact_supply_and_pure()

    # 返回按序排列的dataset
    dataset = load_data()
    fact, ne2qid = fact_build(dataset)  # dict. answer(qid): fact
    neg(dataset, fact)
    # search_fuzz()


if __name__ == "__main__":
    main()
