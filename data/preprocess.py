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


category3_1 = dict()
category3_2 = dict()
category3_3 = dict()
with open('category_names.csv', newline='') as csvfile:
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
        i+=1
print("{} level 1 category".format(len(level1)))
print("{} level 2 category".format(len(level2)))
print("{} level 3 category".format(i+1))

# print(category3_1)
# print(category3_2)
# print(category3_3)

def write_file(file_name, dictionary):
    with open(file_name, 'wb') as fp:
        pickle.dump(dictionary, fp)
        print("label dictionary saved to {}".format(file_name))

def read_file(file_name):
    with open(file_name, 'rb') as fp:
        data = pickle.load(fp)
        print("label dictionary read from {}".format(file_name))
        return data


def extract_img_label(raw_file, outdir, category3_1=None, category3_2=None):
    
    data = bson.decode_file_iter(open(raw_file, 'rb'))
    label_dict = dict()
    prod_to_category = dict()
    idx = 0
    subsize = 3000000
    for c, d in enumerate(data):
        print(idx)
        product_id = d['_id']
        category_id = d['category_id']
        prod_to_category[product_id] = category_id
        # print(type(category_id))
        # print(category3_3[str(category_id)])
#           print('product id is {} category id is {}'.format(product_id, category_id))
        for e, pic in enumerate(d['imgs']):
            if (idx >= 0):
                int_idx = int(idx/subsize)
                dir_path = outdir + "_"+ str(int_idx)
                # print(dir_path)
                os.makedirs(dir_path, exist_ok=True)
                picture = imread(io.BytesIO(pic['picture']))
                name = str(product_id)+"_"+str(e) 
                sublevel_dict = dict()
                if (category3_1 is not None):
                    sublevel_dict[1] = category3_1[str(category_id)]
                if (category3_2 is not None):
                    sublevel_dict[2] = category3_2[str(category_id)]
            
                # sublevel_dict[3] = category_id
                sublevel_dict[3] = category3_3[str(category_id)]
                folder_name = outdir + "_" + str(int_idx) + "/" + name
                label_dict[folder_name] = sublevel_dict
                name = dir_path +"/"+name + ".jpg"
                cv2.imwrite(name, picture)
                if (idx%subsize == (subsize-1)):
            	    write_file(outdir+"_" + str(int_idx) +"_label.p", label_dict)
            	    label_dict = {}
            idx += 1
    if (len(label_dict)!=0):
    	write_file(outdir+"_" + str(int(idx/subsize)) +"_label.p", label_dict)
    print("image files saved to dir ../{}/".format(outdir))


if __name__ == '__main__':
    print("processing and saving images and labels")
    extract_img_label("/media/coffeepanda/Seagate Backup Plus Drive/train.bson", "train", category3_1=category3_1, category3_2=category3_2)


