from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.spatial.distance import pdist, cdist, squareform
from tqdm import tqdm
import models.resnet
import torchvision
import torchvision.transforms as transforms
from custom_adversarial_dataset import AdversarialDataset



batch_size = 1
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
trainset = torchvision.datasets.CIFAR10(root='./data/raw_data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)
testset = torchvision.datasets.CIFAR10(root='./data/raw_data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)


transform = transforms.Compose([ transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
test_data = AdversarialDataset("custom_data/cifar_10_resnet_32_fgsm/cifar_10_resnet_32_fgsm_test/mapping.csv", "custom_data/cifar_10_resnet_32_fgsm/cifar_10_resnet_32_fgsm_test", transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)


net = models.resnet.resnet32()
net.load_state_dict(torch.load('cifar10_resnet32-cifar10_best.pth'))
net = net.cuda()

correct = 0
total = 0
net.eval()

with torch.no_grad():
    i = -1
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        i+=1
        if i % 2 == 0:
            continue
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        print("i: ", i, " pred:", predicted.item())
        print("i: ", i, " targ:", targets.item())
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()   

print("acc: ", 100.0*correct/total)