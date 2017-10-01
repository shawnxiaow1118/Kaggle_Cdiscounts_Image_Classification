import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np 
from PIL import Image
from torch.utils.data import DataLoader
from utils import *
import random

DATA_PATH = "../data/"

class ClassifySet(data.Dataset):
	def __init__(self, data, transforms):
		super(ClassifySet, self).__init__()
		self.image_names = list(data.keys())
		self.l1 = []
		self.l2 = []
		self.l3 = []
		self.transforms = transforms
		for key in self.image_names:
			self.l1.append(data[key][1])
			self.l2.append(data[key][2])
			self.l3.append(data[key][3])


	def __getitem__(self, index):
		img_name = self.image_names[index]
		level1 = self.l1[index]
		level2 = self.l2[index]
		level3 = self.l3[index]
		img_path = DATA_PATH + img_name + ".jpg"
		image = Image.open(img_path).convert('RGB')
		image = resize(image, (64,64))
		image = image_to_tensor(image, 0, 255)
		if self.transforms is not None:
			for t in self.transforms:
				image = t[image]
		return image, level1, level2, level3

	def __len__(self):
		return len(self.image_names)


def get_loader(data, batch_size, num_workers,shuffle=True, transforms=None):
	cdiscount = ClassifySet(data, transforms)
	data_loader = DataLoader(dataset=cdiscount, batch_size=batch_size,
							shuffle=shuffle, num_workers=num_workers)
	return data_loader 


def generator(data, batch_size):
    data_len = len(data)
    image_names = list(data.keys())
    random.shuffle(image_names)
    for i in range(0,data_len, batch_size):
        start = i
        if start+batch_size > data_len:
            flag = False
        end = min(data_len, start+batch_size)
        out_names = image_names[start: end]
        img_list = []
        l1_list = []
        l2_list = []
        l3_list = []
        for name in out_names:
            img_path = DATA_PATH + name + ".jpg"
            image = Image.open(img_path).convert('RGB')
            image = resize(image, (64,64))
            image = image_to_tensor(image, 0, 255)
            img_list.append(image)
            l1_list.append(data[name][1])
            l2_list.append(data[name][2])
            l3_list.append(data[name][3])
        img = torch.stack(img_list, 0)
        l1 = torch.LongTensor(l1_list)
        l2 = torch.LongTensor(l2_list)
        l3 = torch.LongTensor(l3_list)
        yield img, l1, l2, l3