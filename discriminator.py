from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn.utils import spectral_norm

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ESRDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(ESRDiscriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 2), int(in_width / 2 ** 2)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            # if not first_block:
            #     layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            #layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        #for i, out_filters in enumerate([64, 128, 256, 512]):
        for i, out_filters in enumerate([16, 32]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        #layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.classifier = nn.Sequential(
            nn.Linear(32 * self.output_shape[1] * self.output_shape[2], 1),
            #nn.Linear(128 * self.output_shape[1] * self.output_shape[2], 32),
            #nn.LeakyReLU(0.2, True),
            #nn.Linear(32, 1),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        out = self.model(img)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class LSDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(LSDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=False):
            #block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(input_shape[0], 16, bn=False),
            *discriminator_block(16, 32, bn=False),
            *discriminator_block(32, 64, bn=False),
            *discriminator_block(64, 128, bn=False),
        )

        # The height and width of downsampled image
        ds_size1 = input_shape[1] // 2 ** 4
        ds_size2 = input_shape[2] // 2 ** 4
        self.adv_layer = nn.Linear(128 * ds_size1 * ds_size2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
class BasicDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(BasicDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(input_shape[0], 16, bn=False),
        )

        # The height and width of downsampled image
        ds_size1 = input_shape[1] // 2 ** 1
        ds_size2 = input_shape[2] // 2 ** 1
        self.adv_layer = nn.Linear(16 * ds_size1 * ds_size2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
class DCDiscriminator(nn.Module):
    ''' DC Discriminator class.
    Args:
        in_dim (int): input dimension
        n_feat (int): features of final hidden layer
        img_size (int): input image size
    '''
    def __init__(self, in_dim=3, n_feat=512, img_size=32):
        super(DCDiscriminator, self).__init__()

        self.in_dim = in_dim
        n_layers = int(log2(img_size) - 2)
        self.blocks = nn.ModuleList(
            [nn.Conv2d(
                in_dim,
                int(n_feat / (2 ** (n_layers - 1))),
                4, 2, 1, bias=False)] + [nn.Conv2d(
                    int(n_feat / (2 ** (n_layers - i))),
                    int(n_feat / (2 ** (n_layers - 1 - i))),
                    4, 2, 1, bias=False) for i in range(1, n_layers)])

        self.conv_out = nn.Conv2d(n_feat, 1, 4, 1, 0, bias=False)
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        if x.shape[1] != self.in_dim:
            x = x[:, :self.in_dim]
        for layer in self.blocks:
            x = self.actvn(layer(x))

        out = self.conv_out(x)
        out = out.reshape(batch_size, 1)
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ESRDiscriminator(input_shape=[3,32,32]).to(device=device)
    #net = DCDiscriminator(img_size=36, n_feat=128).to(device=device)
    #net = BasicDiscriminator(input_shape=[3,32,32]).to(device=device)
    #net = LSDiscriminator(input_shape=[3,32,32]).to(device=device)
    print("Input(=image) : ")
    #print(torch.randn(1,3,256,64).size())
    y = net(Variable(torch.randn(1,3,32,32)).to(device)) # Input should be a 4D tensor

    from torchsummary import summary
    print(summary(net, (3,32,32)))
    #print("Output(batchsize, channels, width, height) : ")
    print(y.shape)
    print(y)