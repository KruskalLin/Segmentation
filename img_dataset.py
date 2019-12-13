import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import numpy as np
import cv2

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def onehot(data, n):
    buf = np.zeros(data.shape + (n,))
    nmsk = np.arange(data.size) * n + data.ravel()
    buf.ravel()[nmsk - 1] = 1
    return buf


class ImgDataset(Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root + '/image_2'))

    def __getitem__(self, idx):
        img_name = os.listdir(self.root + '/image_2')[idx]
        imgA = cv2.imread(self.root + '/image_2/' + img_name)
        imgA = cv2.resize(imgA, (512, 256))
        imgB = cv2.imread(self.root + '/gt_image_gray/' + img_name, 0)
        imgB = cv2.resize(imgB, (512, 256))
        imgC = cv2.imread(self.root + '/lidar/' + img_name)
        imgC = cv2.resize(imgC, (512, 256))

        img_segs = img_name.split('_')
        imgD_path = self.root + '/gt_image_2/' + img_segs[0] + '_road_' + img_segs[1]

        imgB = imgB / 255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2)
        imgB = imgB.transpose((2, 0, 1))
        imgB = torch.FloatTensor(imgB)
        if self.transform:
            imgA = self.transform(imgA)
            imgC = self.transform(imgC)
        return imgA, imgC, imgB, imgD_path


train = './data/object/training'
dataset = ImgDataset(train, transform)
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

if __name__ == '__main__':

    for train_batch in train_dataloader:
        print(train_batch)

    for test_batch in test_dataloader:
        print(test_batch)
