import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
from deepface import DeepFace
import json
import os
import sys
from tqdm import tqdm
import os

imgData_path = "../../../data/ImgData/wikipedia"
output_path = "../../../data/wikipedia/"

ins = instanceSegmentation()
ins.load_model("../../../data/pretrain_models/pointrend_resnet50.pkl")
target_classes = ins.select_target_classes(person=True)

img_list = os.listdir(imgData_path)

seg_out_path = os.path.join(output_path, "wiki_segement")
exist_files =os.listdir(seg_out_path)

for img_name in tqdm(img_list):
    img_id = img_name.split(".")[0]
    img_path = os.path.join(imgData_path, img_name)

    if str(img_id)+'_1.jpg' in exist_files:
        continue
    try:
        r, output = ins.segmentImage(img_path, show_bboxes=True, extract_segmented_objects=True,
                                 segment_target_classes=target_classes, save_extracted_objects=True, output_image_name=None, output_path=seg_out_path, img_id=img_id)
    except:
        print("Exception ", img_id)
    # print(r['extracted_objects'])
    # print(r["object_counts"])

    objects_num = r["object_counts"]["person"]
    for i in range(1, objects_num + 1):
        detection_id = str(img_id) + '_' + str(i)
        input_name = detection_id + '.jpg'
        input_path = os.path.join(output_path, "wiki_segement", input_name)
        detection_name = "{}.json".format(detection_id)
        detection_path = os.path.join(output_path, "wiki_detection", detection_name)

        objs = DeepFace.analyze(img_path=input_path, actions=['age', 'gender', 'race', 'emotion'],
                                enforce_detection=False, silent=True)
        json.dump(objs, open(detection_path, 'w'), indent=2)

# docker run -itd --name deepface -v E:\Season\MyCode\MM\Face\deepface:/home/Face/deepface b291dda7ce8d /bin /bash
