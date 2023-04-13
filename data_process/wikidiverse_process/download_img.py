import json
import hashlib
import re
import os
import shutil
from util import svg2png, gif2png
from tqdm import tqdm
from urllib.parse import unquote
# ../../../data/
wikipedia_path = "/data/data/Wiki/Wikipedia"
mention_images_dir = "/data/data/Wiki/wikinewsImgs/"
train_dataset_path = "/data/data/Wiki/wikidiverse_w_cands/train_w_10cands.json"
dev_dataset_path = "/data/data/Wiki/wikidiverse_w_cands/valid_w_10cands.json"
test_dataset_path = "/data/data/Wiki/wikidiverse_w_cands/test_w_10cands.json"

# 处理后的图片保存位置
target_images_dir = './data/wikidiverse/images/'
if not os.path.exists(target_images_dir):
    os.makedirs(target_images_dir)
target_dataset_dir = './data/wikidiverse/dataset/'
if not os.path.exists(target_dataset_dir):
    os.makedirs(target_dataset_dir)
entity_embedding_cache = 'data/wikidiverse/candidate_entity_embeddings.json'


# wikidiverse数据集只提供了提及图片的压缩包和实体图片的下载链接
# 为方便处理，本文提前下载了实体图片，并以“链接md5编码+图片格式”的命名规则进行保存
# 本函数的作用是确保实体与提及图片的命名规则一致
def process_mention_images(mention_images_dir, train_dataset_path, dev_dataset_path, test_dataset_path):
    def process_dataset(dataset_path):
        with open(dataset_path, encoding='utf-8') as file:
            data = json.loads(file.readline())

        missing_image_list = []
        for item in tqdm(data):
            # 根据实体图片链接，生成图片名称
            m_img = item[1].split('/')[-1]
            prefix = hashlib.md5(m_img.encode()).hexdigest()
            suffix = re.sub(r'(\S+(?=\.(jpg|JPG|png|PNG|svg|SVG)))|(\S+(?=\.(jpeg|JPEG)))', '', m_img)
            m_img = os.path.join(mention_images_dir, prefix + suffix)
            old_img_path = m_img.replace('.svg', '.png').replace('.SVG', '.png')

            # 检查图片是否存在
            if not os.path.exists(old_img_path):
                missing_image_list.append(item[1])
                continue

            # 根据实体图片链接，生成新的图片名称，并存放在当前项目下
            m_img = item[1]
            prefix = hashlib.md5(m_img.encode()).hexdigest()
            suffix = re.sub(r'(\S+(?=\.(jpg|JPG|png|PNG|svg|SVG|gif|GIF)))|(\S+(?=\.(jpeg|JPEG)))', '',
                            m_img.split('/')[-1])
            suffix = re.sub(r'((?<=\.(jpg|JPG|png|PNG|svg|SVG|gif|GIF))\S+)|((?<=\.(jpeg|JPEG))\S+)', '', suffix)
            m_img = os.path.join(target_images_dir, prefix + suffix)
            new_image_path = m_img.replace('.svg', '.png').replace('.SVG', '.png')

            # 将图片重命名并移动到目标位置
            shutil.copy(old_img_path, new_image_path)

        return missing_image_list

    missing_image_list = process_dataset(train_dataset_path)
    print(len(missing_image_list))
    missing_image_list = process_dataset(dev_dataset_path)
    print(len(missing_image_list))
    missing_image_list = process_dataset(test_dataset_path)
    print(len(missing_image_list))


# 根据wikidiverse数据集提供的图片链接，下载图片到指定文件夹下
# 文件采用“链接md5编码+图片格式”的命名规则进行保存
def download_entity_image(wikipedia_path, train_dataset_path, dev_dataset_path, test_dataset_path):
    cand_set = set()

    def add_entities(dataset_path):
        with open(dataset_path, encoding='utf-8') as file:
            data = json.loads(file.readline())
            for caption, img, ment, ment_type, lctx, rctx, entity, cands, topic, start, end in data:
                cand_set.add(entity.split('/')[-1])
                for cand in cands:
                    cand_set.add(cand.split('/')[-1])

    add_entities(train_dataset_path)
    add_entities(dev_dataset_path)
    add_entities(test_dataset_path)

    # 用来记录实体与图片的对应关系，避免使用时重复处理
    images_dict = {}
    with open(wikipedia_path, encoding='utf-8') as file:
        for i, line in tqdm(enumerate(file)):
            try:
                entity, img_url_list = line.split('@@@@')
                img_url_list = img_url_list.strip().split('[AND]')
            except Exception:
                print(line)
                continue

            # 判断该实体是否在数据集中出现
            if entity not in cand_set:
                continue

            images_dict[entity] = []

            # 根据url列表下载对应图片
            for img_url in img_url_list:
                m_img = img_url
                prefix = hashlib.md5(m_img.encode()).hexdigest()
                suffix = re.sub(r'(\S+(?=\.(jpg|JPG|png|PNG|svg|SVG|gif|GIF)))|(\S+(?=\.(jpeg|JPEG)))', '',
                                m_img.split('/')[-1])
                suffix = re.sub(r'((?<=\.(jpg|JPG|png|PNG|svg|SVG|gif|GIF))\S+)|((?<=\.(jpeg|JPEG))\S+)', '', suffix)
                image_path = os.path.join(target_images_dir, prefix + suffix)

                # 下载函数，根据情况选择

                if os.path.exists(image_path):
                    if suffix.lower() == '.svg':
                        if not svg2png(os.path.join(target_images_dir, prefix + suffix),
                                       os.path.join(target_images_dir, prefix + '.png')):
                            continue
                        suffix = '.png'
                    elif suffix.lower() == '.gif':
                        gif2png(os.path.join(target_images_dir, prefix + suffix),
                                os.path.join(target_images_dir, prefix + '.png'))
                        suffix = '.png'
                    images_dict[entity].append(prefix + suffix)

            if images_dict[entity] == []:
                del images_dict[entity]

    with open(os.path.join(target_images_dir, 'images_map.json'), 'w', encoding='utf-8') as writer:
        print(json.dumps(images_dict, ensure_ascii=False), file=writer)


# 生成项目中统一格式的数据集
def generate_detaset(origin_dataset_path, new_dataset_path, images_map_file):
    def clean_name(text, is_ment):
        text = text.strip()
        if not is_ment:
            text = text.replace('nhttps://en.wikipedia.org/wiki/', '').replace('hhttps://en.wikipedia.org/wiki/', ''). \
                replace(']https://en.wikipedia.org/wiki/', '').replace('https://en.wikipedia.org/wiki/', '').replace(
                'ttps://en.wikipedia.org/wiki/', '').replace('_', ' ').replace('-', ' ')
        text = unquote(text)
        return text

    with open(images_map_file) as file:
        images_map = json.loads(file.readlines()[0])

    # Wikipedia2vec包无法获取部分实体的嵌入表示，因此在生成数据集前，需要进行排除
    # with open(entity_embedding_cache) as reader:
    #     entity_embeddings = json.load(reader)
    #     candidate_entity_list = entity_embeddings.keys()

    samples = []
    miss_samples_num = 0
    with open(origin_dataset_path, encoding='utf-8') as file:
        data = json.loads(file.readline())
        for caption, img, ment, ment_type, lctx, rctx, entity, cands, topic, start, end in data:
            sample = {}

            sample['sentence'] = caption

            sample['mention'] = ment

            m_img = img
            prefix = hashlib.md5(m_img.encode()).hexdigest()
            suffix = re.sub(r'(\S+(?=\.(jpg|JPG|png|PNG|svg|SVG|gif|GIF)))|(\S+(?=\.(jpeg|JPEG)))', '',
                            m_img.split('/')[-1])
            suffix = re.sub(r'((?<=\.(jpg|JPG|png|PNG|svg|SVG|gif|GIF))\S+)|((?<=\.(jpeg|JPEG))\S+)', '', suffix)
            sample['mention_image'] = (prefix + suffix).replace('.svg', '.png').replace('.SVG', '.png').replace('.GIF',
                                                                                                                '.png').replace(
                '.gif', '.png')

            sample['mention_type'] = ment_type

            sample['entity'] = clean_name(entity, is_ment=False)
            # if sample['entity'] not in candidate_entity_list:
            #     miss_samples_num += 1
            #     continue

            sample['entity_image_list'] = images_map.get(entity.split('/')[-1])

            sample['entity_introduction'] = None

            sample['cands'] = []
            for cand in cands:
                entity_name = clean_name(cand, is_ment=False)
                if entity_name not in candidate_entity_list:
                    continue
                sample['cands'].append((entity_name, images_map.get(cand.split('/')[-1])))
            if len(sample['cands']) == 0:
                miss_samples_num += 1
                continue

            sample['topic'] = topic

            sample['start'] = start

            sample['end'] = end

            samples.append(sample)

    with open(new_dataset_path, 'w', encoding='utf-8') as writer:
        print(json.dumps(samples, ensure_ascii=False), file=writer)

    print(miss_samples_num, len(data), miss_samples_num / len(data))


def get_all_candidate_entity():
    def clean_name(text, is_ment):
        text = text.strip()
        if not is_ment:
            text = text.replace('nhttps://en.wikipedia.org/wiki/', '').replace('hhttps://en.wikipedia.org/wiki/', ''). \
                replace(']https://en.wikipedia.org/wiki/', '').replace('https://en.wikipedia.org/wiki/', '').replace(
                'ttps://en.wikipedia.org/wiki/', '').replace('_', ' ').replace('-', ' ')
        text = unquote(text)
        return text

    def get_cand_entity(dataset_path):
        with open(dataset_path, encoding='utf-8') as file:
            data = json.loads(file.readline())
            for caption, img, ment, ment_type, lctx, rctx, entity, cands, topic, start, end in data:
                for cand in cands:
                    print(clean_name(cand, is_ment=False), file=writer)

    writer = open('wikidiverse_cands.txt', 'w', encoding='utf-8')
    get_cand_entity(train_dataset_path)
    get_cand_entity(dev_dataset_path)
    get_cand_entity(test_dataset_path)
    writer.close()


if __name__ == "__main__":
    # process_mention_images(mention_images_dir, train_dataset_path, dev_dataset_path, test_dataset_path)
    # download_entity_image(wikipedia_path, train_dataset_path, dev_dataset_path, test_dataset_path)
    generate_detaset(train_dataset_path, os.path.join(target_dataset_dir, 'train.json'),
                     os.path.join(target_images_dir, 'images_map.json'))
    generate_detaset(dev_dataset_path, os.path.join(target_dataset_dir, 'dev.json'),
                     os.path.join(target_images_dir, 'images_map.json'))
    generate_detaset(test_dataset_path, os.path.join(target_dataset_dir, 'test.json'),
                     os.path.join(target_images_dir, 'images_map.json'))
    # get_all_candidate_entity()