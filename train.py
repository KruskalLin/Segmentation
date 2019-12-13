from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
import cv2
from img_dataset import test_dataloader, train_dataloader
from FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet, FusionNet
from pspnet import PSPNet
from evaluate import *


def train(epo_num=50, show_vgg_params=False):
    vis = visdom.Visdom()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    fcn_model = PSPNet(n_classes=2)
    fcn_model = fcn_model.to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)

    all_train_iter_loss = []
    all_test_iter_loss = []

    # start timing
    prev_time = datetime.now()
    for epo in range(epo_num):

        train_loss = 0
        fcn_model.train()
        for index, (img, lidar, label, color_label) in enumerate(train_dataloader):
            img = img.to(device)
            lidar = lidar.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output, out_cls = fcn_model(img)
            output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
            loss = criterion(output, label)
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            output_np = output.cpu().detach().numpy().copy()
            output_np = np.argmin(output_np, axis=1)
            bag_msk_np = label.cpu().detach().numpy().copy()
            bag_msk_np = np.argmin(bag_msk_np, axis=1)

            if np.mod(index, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_dataloader), iter_loss))
                vis.images(output_np[:, None, :, :], win='train_pred', opts=dict(title='train prediction'))
                vis.images(bag_msk_np[:, None, :, :], win='train_label', opts=dict(title='label'))
                vis.line(all_train_iter_loss, win='train_iter_loss', opts=dict(title='train iter loss'))

        test_loss = 0
        fcn_model.eval()
        for index, (img, lidar, label, color_label) in enumerate(test_dataloader):

            img = img.to(device)
            lidar = lidar.to(device)
            label = label.to(device)
            with torch.no_grad():
                optimizer.zero_grad()
                output, out_cls = fcn_model(img)
                output = torch.sigmoid(output)
                loss = criterion(output, label)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy()
                output_np = np.argmin(output_np, axis=1)
                bag_msk_np = label.cpu().detach().numpy().copy()
                bag_msk_np = np.argmin(bag_msk_np, axis=1)

                if np.mod(index, 15) == 0:
                    print(r'Testing... Open http://localhost:8097/ to see test result.')
                    vis.images(output_np[:, None, :, :], win='test_pred', opts=dict(title='test prediction'))
                    vis.images(bag_msk_np[:, None, :, :], win='test_label', opts=dict(title='label'))
                    vis.line(all_test_iter_loss, win='test_iter_loss', opts=dict(title='test iter loss'))

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('epoch train loss = %f, epoch test loss = %f, %s'
              % (train_loss / len(train_dataloader), test_loss / len(test_dataloader), time_str))

        if np.mod(epo, 5) == 0:
            torch.save(fcn_model, 'checkpoints/fcn_model_{}.pt'.format(epo))
            print('saveing checkpoints/fcn_model_{}.pt'.format(epo))


def predict():
    shutil.rmtree('./origin')
    shutil.rmtree('./predict')
    os.mkdir('origin')
    os.mkdir('predict')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fcn_model = torch.load('./checkpoints/fcn_model_40.pt')
    fcn_model.to(device)
    for index, (img, lidar, label, color_label) in enumerate(test_dataloader):
        img = img.to(device)
        lidar = lidar.to(device)
        label = label.to(device)
        output, out_cls = fcn_model(img)
        output = torch.sigmoid(output)
        output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
        output_np = np.argmin(output_np, axis=1)
        for i in range(0, len(output_np)):
            color_lb = cv2.imread(color_label[i])
            output_lb = 255 - np.asarray(output_np[i, :, :]) * 255
            output_lb = cv2.resize(output_lb.astype('float32'), (color_lb.shape[1], color_lb.shape[0]))
            cv2.imwrite('./origin/' + str(index) + str(i) + '.png', color_lb)
            cv2.imwrite('./predict/' + str(index) + str(i) + '.png', output_lb)


if __name__ == "__main__":
    # train(epo_num=65, show_vgg_params=False)
    predict()
    main('./predict/', './')
