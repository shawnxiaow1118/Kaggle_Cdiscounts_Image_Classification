from PIL import Image
import numpy as np
import torch
from sklearn.cross_validation import train_test_split
import pickle

def resize(image, size):
	""" resize image"""
	assert(len(size)==2)
	image = image.resize(size, Image.ANTIALIAS)
	return image


def normalize(np_image, mean=0, std=1):
	""" normalize image
	"""
	image = (np_image-mean)/std
	return image

def image_to_tensor(image, mean=0, std=1):
	image = np.array(image, np.float32)
	image = normalize(image, mean, std)
	image = image.transpose(2,0,1)
	tensor = torch.from_numpy(image)
	return tensor  


def custom_split(total, test):
    x_train ,x_test = train_test_split(list(total),test_size=0.2)  
    train_dict = dict()
    test_dict = dict()
    for key in x_train:
        train_dict[key] = total[key]
    for key in x_test:
        test_dict[key] = total[key]
    return train_dict, test_dict 

def read_file(file_name):
    with open(file_name, 'rb') as fp:
        data = pickle.load(fp)
        print("label dictionary read from {}".format(file_name))
        return data