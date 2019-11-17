import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import numpy as np
import cv2

def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    return buf

transformImage = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transformLidar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45], std=[0.22])])

class BagDataset(Dataset):

    def __init__(self, transformImage=None, transformLidar=None):
        self.transformImage = transformImage
        self.transformLidar = transformLidar

    def __len__(self):
        return len(os.listdir('./data/object/training/image_2'))

    def __getitem__(self, idx):
        img_name = os.listdir('./data/object/training/image_2')[idx]
        imgA = cv2.imread('./data/object/training/image_2/' + img_name)
        imgA = cv2.resize(imgA, (160, 160))
        imgB = cv2.imread('./data/object/training/gt_image_gray/'+img_name, 0)
        imgB = cv2.resize(imgB, (160, 160))
        imgC = cv2.imread('./data/object/training/lidar/' + img_name, 0)
        imgC = cv2.resize(imgC, (360, 1200))
        imgC = imgC.astype('uint8')
        imgC = imgC[:,:,np.newaxis]
        imgB = imgB/255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2)
        imgB = imgB.transpose(2,0,1)
        imgB = torch.FloatTensor(imgB)
        if self.transformImage:
            imgA = self.transformImage(imgA)
        if self.transformLidar:
            imgC = self.transformLidar(imgC)
        return imgA, imgC, imgB

bag = BagDataset(transformImage, transformLidar)
train_size = int(0.9 * len(bag))
test_size = len(bag) - train_size
train_dataset, test_dataset = random_split(bag, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

if __name__ =='__main__':

    for train_batch in train_dataloader:
        print(train_batch)

    for test_batch in test_dataloader:
        print(test_batch)
