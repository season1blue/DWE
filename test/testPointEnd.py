import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
from deepface import DeepFace
import json
import os
import sys
from tqdm import tqdm
import os

img_name = "509.jpeg"
img_id = img_name.split(".")[0]

ins = instanceSegmentation()
ins.load_model("../../data/pretrain_models/pointrend_resnet50.pkl")
target_classes = ins.select_target_classes(person=True)
r, output = ins.segmentImage(img_name, show_bboxes=True, extract_segmented_objects=True, segment_target_classes = target_classes, save_extracted_objects=True, output_image_name=None, output_path="seg_result")
# print(r['extracted_objects'])
# print(r["object_counts"])

objects_num = r["object_counts"]["person"]
for i in range(1, objects_num+1):
    detection_id = str(img_id) + '_' + str(i)
    input_name = detection_id + '.jpg'

    detection_name = "detection/{}.json".format(detection_id)
    objs = DeepFace.analyze(img_path="seg_result/{}".format(input_name), actions=['age', 'gender', 'race', 'emotion'],
                            enforce_detection=False, silent=True)

    json.dump(objs, open(detection_name, 'w'), indent=2)

# docker run -itd --name deepface -v E:\Season\MyCode\MM\Face\deepface:/home/Face/deepface b291dda7ce8d /bin /bash
