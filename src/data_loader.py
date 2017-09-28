import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np 
from PIL import Image
from torch.utils.data import DataLoader
from utils import *

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


