"""
Created on Sun Oct 25 2018
@author: Kimin Lee
"""
from __future__ import print_function
import argparse
from custom_adversarial_dataset import AdversarialDataset
import torch
import torch.nn as nn
import data_loader
import numpy as np
import models
import os

from torchvision import transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=200, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', default="cifar10", help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--outf', default='./adv_output/', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', default="resnet", help='resnet | densenet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--adv_type', default="FGSM", help='FGSM | BIM | DeepFool | CWL2')
args = parser.parse_args()
print(args)

in_transform = transforms.Compose([]) #  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


def main():
    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'
    if os.path.isdir(args.outf) == False:
        os.mkdir(args.outf)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)

    # check the in-distribution dataset
    args.num_classes = 10
        
    min_pixel = -2.42906570435
    max_pixel = 2.75373125076
    if args.dataset == 'cifar10':
        if args.adv_type == 'FGSM':
            random_noise_size = 0.25 / 4
            
    model = models.resnet32()
    model.load_state_dict(torch.load("./cifar10_resnet32-cifar10_best.pth", map_location=torch.device('cpu')), strict=True)
    model = model.cuda()

    print('load target data: ', args.dataset)
    _, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, args.dataroot)



    
    adv_data_tot, clean_data_tot, noisy_data_tot = 0, 0, 0
    label_tot = 0
    
    correct, adv_correct, noise_correct = 0, 0, 0
    total, generated_noise = 0, 0

    selected_list = []
    selected_index = 0

    for idx in range(0, len(test_loader.dataset), 2):
        data, target = test_loader.dataset[idx+1]
        adv_data, adv_target = test_loader.dataset[idx]
        
        # make 4d tensor
        shape = (1, 3, 32, 32)

        data = data.reshape(shape)
        adv_data = adv_data.reshape(shape)

        data, target = data.cuda(), target.cuda()
        adv_data, adv_target = adv_data.cuda(), adv_target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        output_clean = model(Variable(adv_data, volatile=True))
        output_adv = model(Variable(adv_data, volatile=True)).max(1)[1]

        noisy_data = torch.add(data, random_noise_size, torch.randn(data.size()).cuda()) 
        noisy_data = torch.clamp(noisy_data, min_pixel, max_pixel)

        if total == 0:
            clean_data_tot = data.clone().cpu()
            label_tot = target.clone().cpu()
            noisy_data_tot = noisy_data.clone().cpu()
        elif total == 1:
            clean_data_tot = torch.stack((clean_data_tot, data.clone().data.cpu()),0)
            label_tot = torch.stack((label_tot, target.clone().cpu()), 0)
            noisy_data_tot = torch.stack((noisy_data_tot, noisy_data.clone().cpu()),0)
            clean_data_tot = clean_data_tot.reshape((-1, 3, 32, 32))
            noisy_data_tot = noisy_data_tot.reshape((-1, 3, 32, 32)) 
        else:
            clean_data_tot = torch.cat((clean_data_tot, data.clone().cpu()),0)
            label_tot = torch.cat((label_tot, target.clone().cpu().reshape(1)),0)
            noisy_data_tot = torch.cat((noisy_data_tot, noisy_data.clone().cpu()),0)        

        print("idx: ", idx,  "clean_data_tot: ", clean_data_tot.shape)

        # torch.clamp(adv_data, min_pixel, max_pixel)
        
        # measure the noise 
        temp_noise_max = torch.abs((data - adv_data).reshape(adv_data.size(0), -1))
        temp_noise_max, _ = torch.max(temp_noise_max, dim=1)
        generated_noise += torch.sum(temp_noise_max)

        if total == 0:
            flag = 1
            adv_data_tot = adv_data.clone().cpu()
        else:
            adv_data_tot = torch.cat((adv_data_tot, adv_data.clone().cpu()),0)

        pred_clean = output_clean.max(1)[1]
        print("Model pred: ", pred_clean)
        print("target: value: ", target)
        equal_flag = pred_clean.eq(target).cpu()

        # compute the accuracy
        # print("Output: ", output_adv)
        pred_adv = output_adv.max(0)[1]
        print("Model adv pred: ", pred_adv)
        equal_flag_adv = pred_adv.eq(target.data).cpu()
        adv_correct += equal_flag_adv.sum()
        
        output = model(Variable(noisy_data, volatile=True))
        # compute the accuracy
        pred = output.data.max(0)[1]
        equal_flag_noise = pred.eq(target.data).cpu()
        noise_correct += equal_flag_noise.sum()
        
        # print("equal_flag: ", equal_flag)
        # print("equal_flag_noise: ", equal_flag_noise)
        # print("equal_flag_adv: ", equal_flag_adv)
        if equal_flag and not equal_flag_adv:
            print("Appending: ", idx)
            selected_list.append(selected_index)
        selected_index += 1
            
        total += data.size(0)

    selected_list = torch.LongTensor(selected_list)
    # print("selected_list: ", selected_list)
    # print("clean_data_tot: ", clean_data_tot.shape)
    clean_data_tot = torch.index_select(clean_data_tot, 0, selected_list)
    # print("clean_data_tot: ", clean_data_tot.shape)
    adv_data_tot = torch.index_select(adv_data_tot, 0, selected_list)
    noisy_data_tot = torch.index_select(noisy_data_tot, 0, selected_list)
    label_tot = torch.index_select(label_tot, 0, selected_list)

    torch.save(clean_data_tot, '%s/clean_data_%s_%s_%s.pth' % (args.outf, args.net_type, args.dataset, args.adv_type))
    torch.save(adv_data_tot, '%s/adv_data_%s_%s_%s.pth' % (args.outf, args.net_type, args.dataset, args.adv_type))
    torch.save(noisy_data_tot, '%s/noisy_data_%s_%s_%s.pth' % (args.outf, args.net_type, args.dataset, args.adv_type))
    torch.save(label_tot, '%s/label_%s_%s_%s.pth' % (args.outf, args.net_type, args.dataset, args.adv_type))

    print('Adversarial Noise:({:.2f})\n'.format(generated_noise / total))
    print('Final Accuracy: {}/{} ({:.2f}%)\n'.format(correct, total, 100. * correct / total))
    print('Adversarial Accuracy: {}/{} ({:.2f}%)\n'.format(adv_correct, total, 100. * adv_correct / total))
    print('Noisy Accuracy: {}/{} ({:.2f}%)\n'.format(noise_correct, total, 100. * noise_correct / total))
    
if __name__ == '__main__':
    main()
