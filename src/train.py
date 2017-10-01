import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data_loader import *
from utils import *
from model import *
import os
import numpy as np

num_epoches = 5
batch_size = 128
learning_rate = 0.0001
save_step = 10000

# load data and dataloader
data = read_file("../data/train_0_label.p")
# train_data, test_data = custom_split(data, 0.005)
# train_loader = data_loader.get_loader(train_data, batch_size, 0)
# test_loader = data_loader.get_loader(test_data, batch_size, 0)

train_generator = generator(data, batch_size)

m_model = model_vgg16(49, 483, 5272)
m_model.cuda()
print(m_model)
optimizer = torch.optim.Adam(m_model.parameters(), lr=learning_rate)

level1 = np.ones(49)
level2 = np.ones(483)
level3 = np.ones(5272)

# for key in data.keys():
# 	level1[data[key][1]] += 1
# 	level2[data[key][2]] += 1
# 	level3[data[key][3]] += 1

# weight1 = cal_weights(level1)
# weight2 = cal_weights(level2)
# weight3 = cal_weights(level3)

criterion1 = nn.CrossEntropyLoss(torch.cuda.FloatTensor(level1))
criterion2 = nn.CrossEntropyLoss(torch.cuda.FloatTensor(level2))
criterion3 = nn.CrossEntropyLoss(torch.cuda.FloatTensor(level3))

# print(weight1)

import time
for epoch in range(num_epoches):
	# cur = time.time()
	i = 0
	for img, l1, l2, l3 in train_generator:
		# xx = time.time()-cur
		image = Variable(img).cuda()
		l1 = Variable(l1).cuda()
		l2 = Variable(l2).cuda()
		l3 = Variable(l3).cuda()

		optimizer.zero_grad()

		l1_pred, l2_pred, l3_pred = m_model(image)
		loss3 = criterion3(l3_pred, l3)
		loss = loss3 + criterion2(l2_pred, l2) + criterion1(l1_pred, l1)
		loss.backward()
		optimizer.step()
		# print(time.time()-xx-cur)
		# print(l3)
		# print(type(l3))
		# print(l3_pred.data)
		# print(type(l3_pred))

		if (i+1)%100==0:
			print("Epoch {} iter {} total_loss {}".format(epoch, i, loss))
			_, pred = torch.max(l3_pred.data, 1)
			correct = (pred==l3.data).sum()
			print("Accuracy: {}".format((100.0*correct)/batch_size))

		if (i+1)%save_step == 0:
			print("model_saved")
			torch.save(m_model.state_dict(), os.path.join('./','model-{}-{}.pkl'.format(epoch, i)))
		i += 1

m_model.eval()
correct = 0
total = 0
for img, l1, l2, l3 in test_loader:
	img = Variable(img).cuda()
	o1, o2, o3 = m_model(img)
	pred = torch.max(o3.data, 1)
	correct += (pred.cpu()==l3).sum()
	total += l3.size(0)

print("Test Accuracy: {}".format((100.0*correct)/total))