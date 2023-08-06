import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
from deepface import DeepFace
import json
import os
import sys
from tqdm import tqdm
import os

imgData_path = "../../../data/ImgData/richpedia/images"
output_path = "../../../data/richpedia/"

ins = instanceSegmentation()
ins.load_model("../../../data/pretrain_models/pointrend_resnet50.pkl")
target_classes = ins.select_target_classes(person=True)

img_list = os.listdir(imgData_path)

seg_out_path = os.path.join(output_path, "rich_segement")

for img_name in tqdm(img_list):
    img_id = img_name.split("_")[0]
    img_path = os.path.join(imgData_path, img_name)

    # 已经有图片被detect过了，保存detection.json文件了
    if str(img_id) + '.json' in os.listdir(os.path.join(output_path, "rich_detection")):
        continue
    try:
        r, output = ins.segmentImage(img_path, show_bboxes=True, extract_segmented_objects=True,
                                     segment_target_classes=target_classes, save_extracted_objects=True,
                                     output_image_name=None, output_path=seg_out_path, img_id=img_id)
        objs = DeepFace.analyze(img_path=img_path, actions=['age', 'gender', 'race', 'emotion'],
                                enforce_detection=False, silent=True)
        detection_name = "{}.json".format(img_id)
        detection_path = os.path.join(output_path, "rich_detection", detection_name)
        json.dump(objs, open(detection_path, 'w'), indent=2)

    except Exception as e:
        print("Exception ", e, img_id)





# docker run -itd --name deepface -v E:\Season\MyCode\MM\Face\deepface:/home/Face/deepface b291dda7ce8d /bin /bash
