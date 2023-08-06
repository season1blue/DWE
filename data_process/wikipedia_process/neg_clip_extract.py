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

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
        """
        self.guid = guid
        self.words = words


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask


def read_examples_from_file(args):
    file_path = args.path_facts
    examples = []
    data = json.load(open(file_path, encoding="utf-8"))
    keys_ordered = sorted(list(data.keys()))
    json.dump(keys_ordered, open(os.path.join(args.dir_output, "neg_list.json"), 'w'), indent=2)
    for key in keys_ordered:
        examples.append(InputExample(guid=key, words=data[key]))
    return examples


def convert_examples_to_features(
        examples,
        max_seq_length,
        tokenizer,
):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index > 2:
            break
        print(example.words)
        sequence_dict = tokenizer.encode_plus(example.words, max_length=max_seq_length, pad_to_max_length=True)
        input_ids = sequence_dict['input_ids']
        input_mask = sequence_dict['attention_mask']
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask)
        )
    return features


def load_and_cache_examples(args, tokenizer, mode):
    # logger.info("Creating features from dataset file at %s", args.data_dir)
    examples = read_examples_from_file(args)
    features = convert_examples_to_features(
        examples,
        args.max_seq_length,
        tokenizer,
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask)
    return dataset

def gen_gt_feats(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training

    if args.gt_type == "brief":
        file_path = args.path_brief
    else:
        file_path = args.path_property

    data = json.load(open(file_path, encoding="utf-8"))
    keys_ordered = sorted(list(data.keys()))
    json.dump(keys_ordered, open(os.path.join(args.dir_output, "neg_list.json"), 'w'), indent=2)
    sen_embeddings = torch.FloatTensor(len(keys_ordered), 512)
    model.eval()
    model.to(device)
    for index, key in tqdm(enumerate(keys_ordered), total=len(keys_ordered)):
        input_sent = data[key]
        input_sent = input_sent.split("[SEP]. ")[-1]
        sent_ids = clip.tokenize(input_sent, truncate=True).to(device)
        with torch.no_grad():
            text_feature = model.encode_text(sent_ids)
            sen_embeddings[index] = text_feature

    filename = "gt_feats_{}.h5".format(args.gt_type)
    h5_file = h5py.File(os.path.join(args.dir_output, filename), 'w')
    h5_file.create_dataset("features", data=sen_embeddings.numpy())
    return sen_embeddings




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_brief", default="../../data/wikipedia/brief.json", type=str)
    parser.add_argument("--path_property", default="../../data/wikipedia/property.json", type=str)
    parser.add_argument("--gt_type", default="brief", type=str)
    parser.add_argument("--dir_output", default="../../data/wikipedia", type=str)
    args = parser.parse_args()

    gen_gt_feats(args)


if __name__ == "__main__":
    main()

