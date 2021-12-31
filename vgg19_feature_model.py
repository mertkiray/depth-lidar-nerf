import imageio
import torch
from matplotlib import pyplot as plt
from torchvision import models, transforms, utils


class Vgg19(torch.nn.Module):
    def __init__(self, layers_to_use=None, requires_grad=False):
        super(Vgg19, self).__init__()

        if layers_to_use is None:
            print('VGG LAYERS CANNOT BE EMPTY PLEASE SET IT IN CONFIG TXT')
            exit(-1)

        self.layers = {'0': 'conv1_1',
                  '2': 'conv1_2',
                  '5': 'conv2_1',
                  '7': 'conv2_2',
                  '10': 'conv3_1',
                  '12': 'conv3_2',
                  '14': 'conv3_3',
                  '16': 'conv3_4',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '23': 'conv4_3',
                  '25': 'conv4_4',
                  '28': 'conv5_1',
                  '30': 'conv5_2',
                  '32': 'conv5_3',
                  '34': 'conv5_4',
                       }

        last_layer = 0
        for layer_num, layer_name in self.layers.items():
            if layer_name == layers_to_use[-1]:
                last_layer = layer_num
        last_layer = int(last_layer)+1

        self.vgg_pretrained_features = models.vgg19(pretrained=True).features[:last_layer].eval()
        print(f'VGG MODEL: {self.vgg_pretrained_features}')
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        # for name, layer in self.vgg_pretrained_features._modules.items():
        #     if isinstance(layer, nn.ReLU):
        #         self.vgg_pretrained_features._modules[name] = nn.ReLU(inplace=False)


    def forward(self, x):
        features = {}
        for name, layer in self.vgg_pretrained_features._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x

        return features

def prepare_images_vgg19(image_batch):
    normalize = transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
    normalized_images = normalize(image_batch)
    return normalized_images


def unnormalize_image(image_batch):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

    unnormalized_image = unnormalize(image_batch)
    return unnormalized_image


def visualize_features(features):
    fig, axarr = plt.subplots(4, 4)
    features_viz = features.cpu().numpy()
    features_viz = features_viz[0]
    features_viz = features_viz - features_viz.min()
    features_viz = features_viz / features_viz.max()
    index = 0
    for i in range(4):
        for j in range(4):
            axarr[i][j].imshow(features_viz[index,:,:], cmap='gray')
            index = index + 1
    plt.show()


if __name__ == '__main__':
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    vgg = Vgg19()
    print(vgg)

    # colorImage = np.array(Image.open('train_data/images/0000005930.png'))
    colorImage = imread('train_data/images/0000005940.png') / 255.
    #colorImage = imread('content.jpg')/255.

    color_image_tensor = colorImage[None, ...]
    color_image_tensor = torch.Tensor(color_image_tensor)
    color_image_tensor = color_image_tensor.permute(0, 3,1,2)
    normalized_color_image = prepare_images_vgg19(color_image_tensor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    features = vgg(normalized_color_image)
    feature1_1 = features['conv1_1'].to(device)
    feature1_2 = features['conv1_2'].to(device)
    feature2_1 = features['conv2_1'].to(device)
    feature2_2 = features['conv2_2'].to(device)
    feature3_4 = features['conv3_4'].to(device)
    feature4_4 = features['conv4_4'].to(device)
    feature5_4 = features['conv5_4'].to(device)

    normalized_color_image = normalized_color_image.to(device)
    color_image_tensor = color_image_tensor.to(device)

    a = unnormalize_image(normalized_color_image)
    a = a.permute(0, 2, 3, 1)
    plt.imshow(a[0].cpu())
    plt.show()
    visualize_features(feature1_1)
    #exit(0)


    #visualize_features(feature3_1)
    #exit(0)
    #TO VISUALIZE
    # a = unnormalize_image(input_img)
    # a = a.permute(0, 2, 3, 1)
    # plt.imshow(a[0])
    # plt.show()

    input_img = torch.ones(normalized_color_image.data.size())
    #input_img = imread('style_input.png') / 255.
    #input_img = input_img[None, ...]
    #input_img = torch.Tensor(input_img)
    #input_img = prepare_images_vgg19(input_img).contiguous()

    input_img = input_img.to(device)
    optimizer = torch.optim.Adam([input_img], lr=5e-4)
    vgg = vgg.to(device)
    input_img.requires_grad_(True)
    vgg.requires_grad_(False)
    loss = torch.nn.MSELoss().to(device)

    run = [0]
    while run[0] <= 100:
        optimizer.zero_grad()

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            feature_input = vgg(input_img)
            #feature_loss = torch.mean((feature1_1 - feature_input['conv1_1']) **2)
            #feature_loss = feature_loss + torch.mean((feature1_2 - feature_input['conv1_2'])**2)
            #feature_loss += torch.mean((feature2_1 - feature_input['conv2_1'])**2)
            #feature_loss = feature_loss + torch.mean((feature2_2 - feature_input['conv2_2'])**2)
            #feature_loss = feature_loss + torch.mean((feature3_1 - feature_input['conv3_1'])**2)
            #feature_loss += torch.mean((feature3_2 - feature_input['conv3_2'])**2)
            #feature_loss += torch.mean((feature3_3 - feature_input['conv3_3'])**2)
            #feature_loss += torch.mean((feature3_4 - feature_input['conv3_4'])**2)
            #feature_loss += torch.mean((feature4_1 - feature_input['conv4_1'])**2)
            #feature_loss = feature_loss + torch.mean((feature4_2 - feature_input['conv4_2'])**2)
            #feature_loss += torch.mean((feature4_3 - feature_input['conv4_3'])**2)
            #feature_loss += torch.mean((feature4_4 - feature_input['conv4_4'])**2)
            #feature_loss = feature_loss + torch.mean((feature5_1 - feature_input['conv5_1'])**2)
            feature_loss = torch.mean((feature1_2 - feature_input['conv1_2'])) * 0.1
            feature_loss = feature_loss + torch.mean((feature2_2 - feature_input['conv2_2'])) * 0.1
            feature_loss = feature_loss + torch.mean((feature3_4 - feature_input['conv3_4']))
            feature_loss = feature_loss + torch.mean((feature4_4 - feature_input['conv4_4']))
            feature_loss = feature_loss + torch.mean((feature5_4 - feature_input['conv5_4']))
            feature_loss.backward()

            run[0] += 1
            if run[0]% 50 == 0 or run[0] == 1:
                print(f'Iteartion {run[0]} Loss : {feature_loss.item()}')
            return feature_loss.item()

        optimizer.step(closure)


    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    print(input_img.shape)
    print(type(input_img))
    input_img = input_img.cpu()
    input_img = input_img.detach()
    input_img = input_img.permute(0, 2,3,1)
    input_img = input_img.numpy()
    input_img = input_img[0]
    print(input_img.shape)
    print(type(input_img))
    plt.imsave('style_image3.png', input_img, cmap='gray')


