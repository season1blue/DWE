from transformers import BertTokenizer, BertModel
import json
import argparse
import logging
import os
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from lavis.models import load_model_and_preprocess
from tqdm import tqdm
import h5py
import clip
from PIL import Image

logger = logging.getLogger(__name__)

def wikipedia_entity(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
                                                                      model_type="pretrain", is_eval=True,
                                                                      device=device)
    vis_processors = vis_processors
    txt_processors = txt_processors

    data = json.load(open(args.path_text, encoding="utf-8"))

    entity_list = []
    text_embeddings = torch.FloatTensor(len(data), 768)
    model.eval()
    model.to(device)
    for index, (entity, desc) in tqdm(enumerate(data.items()), total=len(data), desc="extracting wikipedia entity"):

        sent = desc
        entity_list.append(entity)

        with torch.no_grad():
            sent_ids = txt_processors["eval"](sent)

            text_feature = model.extract_features({"image": None, "text_input": [sent_ids]},
                                                       mode="text").text_embeds[:, 0, :]
            text_embeddings[index] = text_feature

    json.dump(entity_list, open(os.path.join(args.dir_output, "entity_list.json"), 'w'), indent=2)

    text_file = h5py.File(os.path.join(args.dir_output, "text_entity_blip.h5"), 'w')
    text_file.create_dataset("features", data=text_embeddings.numpy())



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_text", default="../../../data/wikipedia/brief.json", type=str)
    parser.add_argument("--dir_output", default="../../../data/wikipedia", type=str)
    args = parser.parse_args()

    wikipedia_entity(args)


if __name__ == "__main__":
    main()

