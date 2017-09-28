import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class model_vgg16(nn.Module):
    def __init__(self, l1, l2, l3):
        super(model_vgg16, self).__init__()
        vgg16 = models.vgg16_bn()
        modules = list(vgg16.children())[0]
        m1 = list(modules.children())[:24]
        m2 = list(modules.children())[24:34]
        m3 = list(modules.children())[34:44]
        self.mod1 = nn.Sequential(*m1)
        self.mod2 = nn.Sequential(*m2)
        self.mod3 = nn.Sequential(*m3)


        self.classifier1 = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, l1),
            )

        self.classifier2 = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, l2),
            )

        self.classifier3 = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, l3),
            )

        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        
    
    def forward(self, images):
        level1 = self.mod1(images)
        level2 = self.mod2(level1)
        level3 = self.mod3(level2)
        
        level1 = level1.view(level1.size(0),-1)
        l1_out = self.classifier1(level1)
        level2 = level2.view(level2.size(0),-1)
        l2_out = self.classifier2(level2)

        level3 = level3.view(level3.size(0),-1)
        l3_out = self.classifier3(level3)

        return l1_out, l2_out, l3_out

        
    def pool(self, channel):
        n_pool = nn.Sequential(
                nn.BatchNorm2d(channel, eps=1e-5, momentum=0.1, affine=True),
                nn.ReLU(inplace),
                nn.MaxPool2d(2)
            )
        return n_pool