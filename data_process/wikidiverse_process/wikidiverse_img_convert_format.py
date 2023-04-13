import os

from utils import svg2png, gif2png
from tqdm import tqdm
import json
# 有一些svg和gif格式的文件，处理后的json文件中映射的是png文件，所以需要将所有用到的图片转成png格式
# 对于raw 图片集需要运行这个，直接下载和处理之后的diverse数据集不需要运行这个

wikidiverse_path = "E:\HIRWorks\data\ImgData\wikidiverse"

train_dataset_path = "E:\HIRWorks\data\wikidiverse\\clean_train.json"
dev_dataset_path = "E:\HIRWorks\data\wikidiverse\\clean_dev.json"
test_dataset_path = "E:\HIRWorks\data\wikidiverse\\clean_test.json"

train = json.load(open(train_dataset_path, "r",encoding='utf-8'))
dev = json.load(open(dev_dataset_path, "r",encoding='utf-8'))
test = json.load(open(test_dataset_path, "r",encoding='utf-8'))

all_data = train + dev + test

mention_image_list = []
for i, item in tqdm(enumerate(all_data), total=len(all_data), ncols=80):
    mention_image_id = item['mention_image'].split(".")[0]
    mention_image_list.append(mention_image_id)

convert_count = 0
all_image_list = os.listdir(wikidiverse_path)
for img in tqdm(all_image_list, total=len(all_image_list)):
    try:
        prefix, suffix = img.split(".")[:2]
        if prefix not in mention_image_list:
            continue
        if suffix.lower() == 'svg':
            svg2png(os.path.join(wikidiverse_path, prefix + "." + suffix),
                    os.path.join(wikidiverse_path, prefix + '.png'))
            convert_count += 1
        elif suffix.lower() == 'gif':
            gif2png(os.path.join(wikidiverse_path, prefix + "." + suffix),
                    os.path.join(wikidiverse_path, prefix + '.png'))
            convert_count += 1

    except Exception as e:
        print(e, img)
        print("convert", convert_count)
print("convert", convert_count)