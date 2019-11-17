from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
import cv2
from imgData import test_dataloader, train_dataloader
from FCN import PFCN, VGGNet


def train(epo_num=50, show_vgg_params=False):

    vis = visdom.Visdom()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    fcn_model = PFCN(pretrained_net=vgg_model, n_class=2)
    fcn_model = fcn_model.to(device)
    criterion = nn.BCELoss().to(device)
    criterion_lidar = nn.BCELoss().to(device)
    criterion_mean = nn.BCELoss().to(device)
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)

    all_train_iter_loss = []
    all_test_iter_loss = []

    # start timing
    prev_time = datetime.now()
    for epo in range(epo_num):
        
        train_loss = 0
        fcn_model.train()
        for index, (img, lidar, label) in enumerate(train_dataloader):
            img = img.to(device)
            lidar = lidar.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output, lidar, mean = fcn_model(img, lidar)
            output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
            lidar_match = torch.sigmoid(lidar)
            mean = torch.sigmoid(mean)
            loss = criterion(output, label)
            loss_lidar = criterion_lidar(lidar_match, label)
            loss_mean = criterion_mean(mean, label)
            loss = loss + loss_lidar + loss_mean
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
            output_np = np.argmin(output_np, axis=1)
            bag_msk_np = label.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 2, 160, 160)
            bag_msk_np = np.argmin(bag_msk_np, axis=1)
            lidar_np = lidar.cpu().detach().numpy().copy()
            lidar_np = np.argmin(lidar_np, axis=1)
            mean_np = lidar.cpu().detach().numpy().copy()
            mean_np = np.argmin(mean_np, axis=1)


            if np.mod(index, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_dataloader), iter_loss))
                # vis.close()
                vis.images(output_np[:, None, :, :], win='train_pred', opts=dict(title='train prediction')) 
                vis.images(bag_msk_np[:, None, :, :], win='train_label', opts=dict(title='label'))
                vis.line(all_train_iter_loss, win='train_iter_loss',opts=dict(title='train iter loss'))
                vis.images(lidar_np[:, None, :, :], win='train_lidar', opts=dict(title='lidar'))
                vis.images(mean_np[:, None, :, :], win='train_mean', opts=dict(title='mean'))

        
        test_loss = 0
        fcn_model.eval()
        for index, (img, lidar, label) in enumerate(test_dataloader):

            img = img.to(device)
            lidar = lidar.to(device)
            label = label.to(device)
            with torch.no_grad():
                optimizer.zero_grad()
                output, lidar, mean = fcn_model(img, lidar)
                output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
                lidar_match = torch.sigmoid(lidar)
                mean = torch.sigmoid(mean)
                loss = criterion(output, label)
                loss_lidar = criterion_lidar(lidar_match, label)
                loss_mean = criterion_mean(mean, label)
                loss = loss + loss_lidar + loss_mean
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)
                output_np = np.argmin(output_np, axis=1)
                bag_msk_np = label.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 2, 160, 160)
                bag_msk_np = np.argmin(bag_msk_np, axis=1)
                lidar_np = lidar.cpu().detach().numpy().copy()
                lidar_np = np.argmin(lidar_np, axis=1)
                mean_np = lidar.cpu().detach().numpy().copy()
                mean_np = np.argmin(mean_np, axis=1)

                if np.mod(index, 15) == 0:
                    print(r'Testing... Open http://localhost:8097/ to see test result.')
                    # vis.close()
                    vis.images(output_np[:, None, :, :], win='test_pred', opts=dict(title='test prediction'))
                    vis.images(bag_msk_np[:, None, :, :], win='test_label', opts=dict(title='label'))
                    vis.line(all_test_iter_loss, win='test_iter_loss', opts=dict(title='test iter loss'))
                    vis.images(lidar_np[:, None, :, :], win='test_lidar', opts=dict(title='lidar'))
                    vis.images(mean_np[:, None, :, :], win='train_mean', opts=dict(title='mean'))


        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('epoch train loss = %f, epoch test loss = %f, %s'
                %(train_loss/len(train_dataloader), test_loss/len(test_dataloader), time_str))
        

        if np.mod(epo, 5) == 0:
            torch.save(fcn_model, 'checkpoints/fcn_model_{}.pt'.format(epo))
            print('saveing checkpoints/fcn_model_{}.pt'.format(epo))


def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg_model = VGGNet(requires_grad=True, show_params=False)
    fcn_model = PFCN(pretrained_net=vgg_model, n_class=2)
    fcn_model = torch.load('./checkpoints/fcn_model_95.pt')
    fcn_model.to(device)
    for index, (img, lidar, label) in enumerate(test_dataloader):
        img = img.to(device)
        lidar = lidar.to(device)
        label = label.to(device)
        output = fcn_model(img, lidar)
        output = torch.sigmoid(output)
        output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
        output_np = np.argmin(output_np, axis=1)
        bag_msk_np = label.cpu().detach().numpy().copy()  # bag_msk_np.shape = (4, 2, 160, 160)
        bag_msk_np = np.argmin(bag_msk_np, axis=1)
        for i in range(0, len(output_np)):
            cv2.imwrite('./origin/' + str(index) + str(i) + '.png', np.asarray(bag_msk_np[i, :, :]) * 255)
            cv2.imwrite('./predict/' + str(index) + str(i) + '.png', np.asarray(output_np[i, :, :]) * 255)




if __name__ == "__main__":
    train(epo_num=100, show_vgg_params=False)
    # predict()
