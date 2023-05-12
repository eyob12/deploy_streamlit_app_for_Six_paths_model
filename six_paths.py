# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:15:19 2023

@author: 1
"""

# -*- coding: utf-8 -*-


import torch
import numpy as numpy
import cv2
import PIL
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm
import torch.nn as nn
import torchvision.transforms as transforms
import imageio
from torch.autograd import Variable
import torch.nn.init as init
import math
import os
import scipy
import json
from scipy import misc,ndimage
import streamlit as st

st.title("Six Paths Testing Model Section")
'Done by Eyob: Six paths based deep learning approch to fuse infrared and visible image'
def singleScaleRetinex(img, variance):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
    return retinex


def SSR(img, variance):
    img = np.float64(img) + 1.0
    img_retinex = singleScaleRetinex(img, variance)
    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)
    return img_retinex


def res_arch_init(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if 'residual' in name:
                init.xavier_uniform_(module.weight, gain=math.sqrt(2))
            else:
                init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                init.zeros_(module.bias)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        if in_channels != out_channels or down:
            shortcut = [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True), ]
        else:
            shortcut = []
        if down:
            shortcut.append(nn.MaxPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)

        residual = [
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        if down:
            residual.append(nn.MaxPool2d(2))
        self.residual = nn.Sequential(*residual)
        res_arch_init(self)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)

        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)

        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )

        self.ResBlock1 = ResBlock(64, 64, down=False)
        self.ResBlock2 = ResBlock(64, 128, down=True)
        self.ResBlock3 = ResBlock(128, 128, down=False)
        self.ResBlock4_1 = ResBlock(128, 256, down=True)

        # self.ResBlock4_2 = ResBlock(256, 256, down=True)

        self.fc1 = nn.Linear(256 * 4 * 4 * 6, 120)
        self.fc2 = nn.Linear(120, 2)

        fc_layer = [nn.Linear(256 * 4 * 4 * 6, 120),
                    nn.BatchNorm1d(120),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Linear(120, 2)]
        self.fc = torch.nn.Sequential(*fc_layer)

    def forward(self, x, y, z, p4, p5, p6):
        # for x
        outx = self.conv1_1(x)
        # print(outx.shape)
        outx = self.ResBlock1(outx)
        # print(outx.shape)
        outx = self.ResBlock2(outx)
        # print(outx.shape)

        outx = self.ResBlock3(outx)
        # print(outx.shape)

        outx = self.ResBlock4_1(outx)
        # print(outx.shape)

        outx = outx.view(outx.size(0), -1)

        ## for y

        outy = self.conv1_1(y)
        # print(outx.shape)
        outy = self.ResBlock1(outy)
        # print(outx.shape)
        outy = self.ResBlock2(outy)
        # print(outx.shape)

        outy = self.ResBlock3(outy)
        # print(outx.shape)

        outy = self.ResBlock4_1(outy)
        # print(outx.shape)

        outy = outy.view(outy.size(0), -1)

        ## for z

        outz = self.conv1_1(z)
        # print(outx.shape)
        outz = self.ResBlock1(outz)
        # print(outx.shape)
        outz = self.ResBlock2(outz)
        # print(outx.shape)

        outz = self.ResBlock3(outz)
        # print(outx.shape)

        outz = self.ResBlock4_1(outz)
        # print(outx.shape)

        outz = outz.view(outz.size(0), -1)

        ## for p4

        outp4 = self.conv1_1(p4)
        # print(outx.shape)
        outp4 = self.ResBlock1(outp4)
        # print(outx.shape)
        outp4 = self.ResBlock2(outp4)
        # print(outx.shape)

        outp4 = self.ResBlock3(outp4)
        # print(outx.shape)

        outp4 = self.ResBlock4_1(outp4)
        # print(outx.shape)

        outp4 = outp4.view(outp4.size(0), -1)

        ## for p5

        outp5 = self.conv1_1(p5)
        # print(outx.shape)
        outp5 = self.ResBlock1(outp5)
        # print(outx.shape)
        outp5 = self.ResBlock2(outp5)
        # print(outx.shape)

        outp5 = self.ResBlock3(outp5)
        # print(outx.shape)

        outp5 = self.ResBlock4_1(outp5)
        # print(outx.shape)

        outp5 = outp5.view(outp5.size(0), -1)

        ## for p5

        outp6 = self.conv1_1(p6)
        # print(outx.shape)
        outp6 = self.ResBlock1(outp6)
        # print(outx.shape)
        outp6 = self.ResBlock2(outp6)
        # print(outx.shape)

        outp6 = self.ResBlock3(outp6)
        # print(outx.shape)

        outp6 = self.ResBlock4_1(outp6)
        # print(outx.shape)

        outp6 = outp6.view(outp6.size(0), -1)



        oyz = torch.cat([outx, outy, outz, outp4, outp5, outp6], 1)

        out = self.fc(oyz)

        return out


model = CNN()
print("model")
model_path = './new_sixpaths_model.pth'

use_gpu = torch.cuda.is_available()

if use_gpu:

    print('GPU Mode Acitavted')
    model = model.cuda()
    # model.cuda()
    state_dict = torch.load(model_path)

    # model.load_state_dict(torch.load(model_path))
    new_state_dict = dict()
    for k, v in state_dict.items():
        name = '.'.join(k.split('.')[1:]) if k.startswith('module') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    # model.load_state_dict({'.'.join(k.split('.')[1:]): v for k, v in check_point.items() if k.split('.')[0] == 'module'})
    model.eval()

else:

    print('CPU Mode Acitavted')
    state_dict = torch.load(model_path, map_location='cpu')

    new_state_dict = dict()
    for k, v in state_dict.items():
        name = '.'.join(k.split('.')[1:]) if k.startswith('module') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    # model.load_state_dict({'.'.join(k.split('.')[1:]): v for k, v in check_point.items() if k.split('.')[0] == 'module'})
    model.eval()


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

@st.cache_data()
def load_s3_file_structure(path: str = 'output.json') -> dict:  # str = 'src/all_image_files.json'
    """Retrieves JSON document outining the S3 file structure"""
    with open(path, 'r') as f:
        return json.load(f)
    
all_image_files = load_s3_file_structure()
index_of_image = sorted(list(all_image_files.keys()))

dataset_type = st.sidebar.selectbox(
            "select the image", index_of_image)

path1 = all_image_files[dataset_type]['org_vis']
path3 = all_image_files[dataset_type]['enh_vis']
path2 = all_image_files[dataset_type]['org_ir']
path4 = all_image_files[dataset_type]['enh_ir']

original_path1 = path1
original_path2 = path2
original_path3 = path3
original_path4 = path4

img3_org = Image.open(original_path3)
img3_org = img3_org.resize((240, 240))
img4_org = Image.open(original_path4)
img4_org = img4_org.resize((240, 240))
img3_org = np.asarray(img3_org)
img4_org = np.asarray(img4_org)

img1_org = Image.open(original_path1)
img1_org = img1_org.resize((240, 240))
img2_org = Image.open(original_path2)
img2_org = img2_org.resize((240, 240))
img1_org = np.asarray(img1_org)
img2_org = np.asarray(img2_org)


col1, col2,col3 = st.columns(3)
with col1:
    st.image(img3_org, caption='Infrared image', use_column_width='auto')
with col2:
    st.image(img1_org, caption='Visible image',use_column_width='auto')
# img4_org = cv2.cvtColor(img4_org, cv2.COLOR_BGR2GRAY)


tfms1 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.45], [0.1])
])

tfms2 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.050], [0.09])
])
tfms3 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.06], [0.09])
])

img1_org = Image.open(original_path1)
img1_org = img1_org.convert('L')
img2_org = Image.open(original_path2)
img2_org = img2_org.convert('L')
img1_org = np.asarray(img1_org)
img2_org = np.asarray(img2_org)
img1_org = cv2.resize(img1_org, (456, 456))
img2_org = cv2.resize(img2_org, (456, 456))
height = img1_org.shape[0]
width = img2_org.shape[1]

img3_org = Image.open(original_path3)
img4_org = Image.open(original_path4)
img3_org = np.asarray(img3_org)
img4_org = np.asarray(img4_org)
img3_org = cv2.resize(img3_org, (456, 456))
img4_org = cv2.resize(img4_org, (456, 456))

windows_size = 32
# stride can be set as 2 or 4 or 8 based on the size of input images
if width >= 500 and height >= 500:
    factor = 2
    stride = 4
else:
    factor = 1
    stride = 8

dim1 = (width, height)
dim2 = (int(width * factor), int(height * factor))
img1 = cv2.resize(img1_org, dim2, interpolation=cv2.INTER_AREA)

img2 = cv2.resize(img2_org, dim2, interpolation=cv2.INTER_AREA)

test_image1_1 = img1
test_image1_2 = img1
test_image1_3 = img1

test_image2_1 = img2
test_image2_2 = img2
test_image2_3 = img2

source1 = img1
source2 = img2

j = 0

MAP = np.zeros([img1.shape[0], img1.shape[1]])

score1 = 0
score2 = 0
FUSED = np.zeros(test_image1_1.shape)

windowsize_r = windows_size - 1
windowsize_c = windows_size - 1

map1 = np.zeros([img1.shape[0], img1.shape[1]])
map2 = np.zeros([img2.shape[0], img2.shape[1]])
img1.shape

for r in tqdm(range(0, img1.shape[0] - windowsize_r, stride)):
    for c in range(0, img1.shape[1] - windowsize_c, stride):

        block_test1_1 = test_image1_1[r:r + windowsize_r + 1, c:c + windowsize_c + 1]
        block_test1_2 = test_image1_2[r:r + windowsize_r + 1, c:c + windowsize_c + 1]
        block_test1_3 = test_image1_3[r:r + windowsize_r + 1, c:c + windowsize_c + 1]

        block_test2_1 = test_image2_1[r:r + windowsize_r + 1, c:c + windowsize_c + 1]
        block_test2_2 = test_image2_2[r:r + windowsize_r + 1, c:c + windowsize_c + 1]
        block_test2_3 = test_image2_3[r:r + windowsize_r + 1, c:c + windowsize_c + 1]

        # block1_1= np.concatenate((block_test1_1, block_test2_1), axis=0)
        # block2_1= np.concatenate((block_test2_1, block_test1_1), axis=0)
        block_test1_1 = Image.fromarray(block_test1_1, 'L')
        block_test1_2 = Image.fromarray(block_test1_2, 'L')
        block_test1_3 = Image.fromarray(block_test1_3, 'L')
        block_test2_1 = Image.fromarray(block_test2_1, 'L')
        block_test2_2 = Image.fromarray(block_test2_2, 'L')
        block_test2_3 = Image.fromarray(block_test2_3, 'L')
        # block2_1 = Image.fromarray(block_test1_2, 'L')
        # block1_2= np.concatenate((block_test1_2, block_test2_2), axis=0)
        # block2_2= np.concatenate((block_test2_2, block_test1_2), axis=0)
        # block1_2 = Image.fromarray(block_test1_3, 'L')
        # block2_2 = Image.fromarray(block_test2_1, 'L')
        # block1_3= np.concatenate((block_test1_3, block_test2_3), axis=0)
        # block2_3= np.concatenate((block_test2_3, block_test1_3), axis=0)
        # block1_3 = Image.fromarray(block_test2_2, 'L')
        # block2_3 = Image.fromarray(block_test2_3, 'L')

        imout1_1 = tfms1(block_test1_1)
        imout2_1 = tfms1(block_test1_2)
        imout1_2 = tfms2(block_test1_3)
        imout2_2 = tfms2(block_test2_1)
        imout1_3 = tfms3(block_test2_2)
        imout2_3 = tfms3(block_test2_3)

        if use_gpu:
            imout1_1 = to_var(imout1_1)
            imout2_1 = to_var(imout2_1)
            imout1_2 = to_var(imout1_2)
            imout2_2 = to_var(imout2_2)
            imout1_3 = to_var(imout1_3)
            imout2_3 = to_var(imout2_3)

        imout1_1 = (imout1_1)
        imout2_1 = (imout2_1)
        imout1_2 = (imout1_2)
        imout2_2 = (imout2_2)
        imout1_3 = (imout1_3)
        imout2_3 = (imout2_3)

        inputs1_1 = imout1_1.unsqueeze(0)
        inputs2_1 = imout2_1.unsqueeze(0)
        inputs1_2 = imout1_2.unsqueeze(0)
        inputs2_2 = imout2_2.unsqueeze(0)
        inputs1_3 = imout1_3.unsqueeze(0)
        inputs2_3 = imout2_3.unsqueeze(0)

        model.eval()

        outputs1 = model(inputs1_1, inputs2_1, inputs1_2, inputs2_2, inputs1_3, inputs2_3)
        _, predicted1 = torch.max(outputs1.data, 1)

        score1 = predicted1.detach().cpu().numpy()

        model.eval()


        if score1 <= 0:
            map1[r:r + windowsize_r + 1, c:c + windowsize_c + 1] += -1

        else:
            map1[r:r + windowsize_r + 1, c:c + windowsize_c + 1] += +1

img3_org = Image.open(original_path3)
img4_org = Image.open(original_path4)

img3_org = np.asarray(img3_org)
img4_org = np.asarray(img4_org)

img3_org = cv2.resize(img3_org, (456, 456))
img4_org = cv2.resize(img4_org, (456, 456))
# img4_org = cv2.cvtColor(img4_org, cv2.COLOR_BGR2GRAY)

# print(np.shape(img4_org))
test_image1 = img1_org
test_image2 = img2_org

map3 = np.zeros(img4_org.shape)
FUSED = np.zeros(img1_org.shape)
distance_map = np.zeros(img1_org.shape)
gmap = np.zeros(img1_org.shape)
weight1 = np.zeros(img1_org.shape)
weight2 = np.zeros(img1_org.shape)

FUSED_8 = np.zeros(map1.shape)
for r in range(0, img1_org.shape[0], 1):
    for c in range(0, img1_org.shape[1], 1):

        if map1[r, c] < 0:
            gmap[r, c] = 0
            map3[r, c] = img4_org[r, c];

        else:
            map3[r, c] = img3_org[r, c];
            gmap[r, c] = 1


weight3 = ndimage.distance_transform_edt(gmap)
weight4 = ndimage.distance_transform_edt((1 - gmap))

FUSED_8 = map3.astype(np.uint8)

distance_map = ndimage.distance_transform_edt(FUSED_8)

first_weight = np.zeros(img1_org.shape)
second_weight = np.zeros(img1_org.shape)

for r in range(0, img1_org.shape[0], 1):
    for c in range(0, img1_org.shape[1], 1):
        first_weight[r, c] = (distance_map[r, c] + weight3[r, c]) / (
                    distance_map[r, c] + distance_map[r, c] + weight3[r, c] + weight4[r, c])
        second_weight[r, c] = (distance_map[r, c] + weight4[r, c]) / (
                    distance_map[r, c] + distance_map[r, c] + weight3[r, c] + weight4[r, c])

        FUSED[r, c] = ((distance_map[r, c] * img4_org[r, c]) + (distance_map[r, c] * img3_org[r, c]) + (
                    weight4[r, c] * img4_org[r, c]) + (weight3[r, c] * img3_org[r, c])) / (
                                  distance_map[r, c] + distance_map[r, c] + weight3[r, c] + weight4[r, c]);

map31 = FUSED.astype(np.uint8)
map31 = cv2.resize(map31, (512, 512))


FUSED = 255 * (FUSED - FUSED.min()) / (FUSED.max() - FUSED.min())

pil_image = Image.fromarray(np.uint8(FUSED))
if pil_image.mode == "F":
    pil_image = pil_image.convert('RGB')
pil_image = pil_image.convert('RGB')

variance = 300
pil_image = SSR(pil_image, variance)

pil_image = cv2.resize(pil_image, (240, 240))

with col3:
    st.image(pil_image, caption='fused Image', use_column_width='auto')




