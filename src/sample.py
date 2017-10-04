import torch
from model import *

import os
import numpy as np
import pandas as pd
import io
import bson
import matplotlib.pyplot as plt
from skimage.data import imread
import multiprocessing as mp
import pickle
import cv2
import csv

from utils import *
from PIL import Image
from torch.autograd import Variable


category3_1 = dict()
category3_2 = dict()
category3_3 = dict()
id_category3 = dict()
with open('../data/category_names.csv', newline='') as csvfile:
    category = csv.reader(csvfile, delimiter=',', quotechar='|')
    level1 = set()
    level2 = set()

    cat1_idx = -1
    cat2_idx = -1

    i = 0
    for row in category:
#         print(row)
        if (i !=0):
            if (row[1] not in level1):
                cat1_idx += 1
                level1.add(row[1])
            category3_1[row[0]] = cat1_idx
            if (row[2] not in level2):
                cat2_idx += 1
                level2.add(row[2])
            category3_2[row[0]] = cat2_idx 
            category3_3[row[0]] = i-1
            id_category3[i-1] = row[0]
        i+=1
print("{} level 1 category".format(len(level1)))
print("{} level 2 category".format(len(level2)))
print("{} level 3 category".format(i+1))

m_model = model_vgg16(49, 483, 5272)
m_model.cuda()

m_model.load_state_dict(torch.load("./model-0-9999.pkl"))
m_model.eval()

data = bson.decode_file_iter(open("../data/test.bson", 'rb'))

i = 0
submit = [["_id","categorical_id"]]
for img in data:
    # if  i > 20:
    #     break
    idx = img['_id']
    i += 1
#     print(img['imgs'][0]['picture'])
    picture = imread(io.BytesIO(img['imgs'][0]['picture']))
#     print(picture)
    # plt.imshow(picture)
    # plt.show()
    img = Image.fromarray(picture, 'RGB')

    image = resize(img, (64,64))
    image = image_to_tensor(image, 0, 255)
    image = image.unsqueeze(0)
    # print(image.size())
    image = Variable(image).cuda()
    l1, l2, l3 = m_model(image)
    _, pred = torch.max(l3.data, 1)
    pred = pred.cpu().numpy()
    submit.append([str(idx),str(id_category3[pred[0]])])
    print("id : {} pred : {}".format(i, pred[0]))

with open("submit.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in submit:
            writer.writerow(line)