import random

import cv2
import os
import glob
import numpy as np

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def dataset(anno_dir, img_dir):
    img_paths = []
    annos = []
    for anno_file in glob.glob(os.path.join(anno_dir, '*.txt')):
        anno_id = anno_file.split('/')[-1].split('.')[0].split('\\')[-1]

        with open(anno_file, 'r') as f:
            num_of_objs = int(file_len(f.name))

            img_path = os.path.join(img_dir, f'{anno_id}.jpg')
            img = cv2.imread(img_path)
            img_height, img_width, _ = img.shape
            del img

            boxes = []
            for _ in range(num_of_objs):
                obj = f.readline().rstrip().split(' ')
                obj = [float(elm) for elm in obj]
                obj[0] = int(obj[0])
                
                xmin = max(obj[1], 0)
                ymin = max(obj[2], 0)
                xmax = min(obj[3], img_width) 
                ymax = min(obj[4], img_height)

                boxes.append([obj[0], xmin, ymin, xmax, ymax])

            if not boxes:
                continue

        img_paths.append(img_path)
        annos.append(boxes)
    return img_paths, annos