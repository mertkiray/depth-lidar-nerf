from collections import namedtuple

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchvision import models, transforms, utils
from PIL import Image
import torch.nn.functional as F


class Resnet(torch.nn.Module):
    def __init__(self, output_layer):
        super(Resnet, self).__init__()
        self.pretrained = models.resnet18(pretrained=True)
        print(self.pretrained)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1, len(self.layers) - self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])

        self.net = nn.Sequential(self.pretrained._modules)
        self.net.requires_grad_(False)
        self.pretrained = None


    def forward(self, x):
        x = self.net(x)
        return x

def prepare_images_vgg19(image_batch):
    normalize = transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
    #cropped_imamges = crop(image_batch)
    normalized_images = normalize(image_batch)
    return normalized_images


def unnormalize_image(image_batch):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

    unnormalized_image = unnormalize(image_batch)
    return unnormalized_image


def visualize_features(features, title):

    fig, axarr = plt.subplots(4, 4)

    print(features.shape)

    features_viz = features.cpu().numpy()
    features_viz = features_viz[0]
    features_viz = features_viz - features_viz.min()
    features_viz = features_viz / features_viz.max()
    print(features_viz.shape)

    index = 0
    for i in range(4):
        for j in range(4):
            axarr[i][j].imshow(features_viz[index,:,:], cmap='gray')
            index = index + 1

    fig.canvas.set_window_title(title)
    plt.show()


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)


    resnet = Resnet(output_layer='layer1')
    resnet = resnet.to(device)

    from torchsummary import summary

    summary(resnet, input_size=(3, 47, 176))

    # colorImage = np.array(Image.open('train_data/images/0000005930.png'))
    #colorImage = imread('train_data/images/0000005930.png') / 255.
    colorImage = imread('individualImage.png')/255.

    color_image_tensor = colorImage[None, ...]
    color_image_tensor = torch.Tensor(color_image_tensor)
    color_image_tensor = color_image_tensor.to(device)

    color_image_tensor = color_image_tensor.permute(0, 3, 1, 2)


    normalized_color_image = prepare_images_vgg19(color_image_tensor)
    normalized_color_image = normalized_color_image.to(device)


    features = resnet(normalized_color_image)
    features = features.to(device)


    a = unnormalize_image(normalized_color_image)
    a = a.permute(0, 2, 3, 1)
    plt.imshow(a[0].cpu())
    plt.show()


    #visualize_features(features, 'features')
    #exit(0)


    #visualize_features(feature3_1)
    #exit(0)
    #TO VISUALIZE
    # a = unnormalize_image(input_img)
    # a = a.permute(0, 2, 3, 1)
    # plt.imshow(a[0])
    # plt.show()

    input_img = torch.zeros(normalized_color_image.data.size())

    #input_img = imread('style_input.png') / 255.
    #input_img = input_img[None, ...]
    #input_img = torch.Tensor(input_img)
    #input_img = input_img.permute(0, 3, 1, 2)
    #input_img = prepare_images_vgg19(input_img).contiguous()

    input_img = input_img.to(device)
    optimizer = torch.optim.Adam([input_img], lr=0.001)
    vgg = resnet.to(device)
    input_img.requires_grad_(True)
    vgg.requires_grad_(False)
    loss = torch.nn.MSELoss().to(device)

    run = [0]
    while run[0] <= 5000:
        optimizer.zero_grad()

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            feature_input = vgg(input_img)
            feature_loss = 0
            feature_loss = feature_loss + torch.mean((features - feature_input)**2)
            feature_loss.backward()

            run[0] += 1
            if run[0]% 50 == 0 or run[0] == 1:
                print(f'Iteartion {run[0]} Loss : {feature_loss.item()}')
            return feature_loss.item()

        optimizer.step(closure)


    #a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    print(input_img.shape)
    print(type(input_img))
    #input_img = unnormalize_image(input_img)
    input_img = input_img.cpu()
    input_img = input_img.detach()
    input_img = input_img.permute(0, 2,3,1)
    input_img = input_img.numpy()
    input_img = input_img[0]
    print(input_img.shape)
    print(type(input_img))
    plt.imsave('style_image4.png', input_img)


