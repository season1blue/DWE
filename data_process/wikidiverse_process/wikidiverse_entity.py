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
from PIL import Image

logger = logging.getLogger(__name__)

def wikiperson_entity(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training

    data = json.load(open(args.path_text, encoding="utf-8"))

    entity_list = []
    text_embeddings = torch.FloatTensor(len(data), 512)
    model.eval()
    model.to(device)
    for index, (entity, desc) in tqdm(enumerate(data.items()), total=len(data), desc="extracting wikidiverse entity"):

        sent = entity + ' [SEP] ' + desc
        entity_list.append(entity)

        sent_ids = clip.tokenize(sent, truncate=True).to(device)
        with torch.no_grad():
            text_feature = model.encode_text(sent_ids).to(device)
            text_embeddings[index] = text_feature

    json.dump(entity_list, open(os.path.join(args.dir_output, "entity_list.json"), 'w'), indent=2)

    text_file = h5py.File(os.path.join(args.dir_output, "text_entity.h5"), 'w')
    text_file.create_dataset("features", data=text_embeddings.numpy())



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_text", default="../../../data/wikidiverse/entity2desc.json", type=str)
    parser.add_argument("--path_image", default="../../../data/wikidiverse/IKB/", type=str)
    parser.add_argument("--dir_output", default="../../../data/wikidiverse", type=str)
    args = parser.parse_args()

    wikiperson_entity(args)


if __name__ == "__main__":
    main()

