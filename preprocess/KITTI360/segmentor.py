import cv2
import imageio
import numpy
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from matplotlib import pyplot as plt

class SemanticSegmentor(object):
    def __init__(self, segmentor_config='../../configs/deeplab/deeplab_pretrained.yaml',
                 segmentor_weights='../../pretrained/deeplab_pretrained.pkl'):

        cfg = get_cfg()
        add_deeplab_config(cfg)
        cfg.merge_from_file(segmentor_config)
        cfg.MODEL.WEIGHTS = segmentor_weights
        print(segmentor_config)
        print(segmentor_weights)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8 # set a custom testing threshold
        cfg.INPUT.CROP.ENABLED = False
        cfg.freeze()
        self.cfg = cfg

        self.predictor = DefaultPredictor(cfg)


    def segment_image(self, image):
        outputs = self.predictor(image)
        outputs = outputs['sem_seg'].cpu()
        return outputs


class SemanticSegmentorHelper(object):
    def __init__(self):
        self.rgbs = np.array([
            [128, 64, 128],  # road             0
            [244, 35, 232],  # sidewalk         1
            [70, 70, 70],  # building           2
            [102, 102, 156],  # wall            3
            [190, 153, 153],  # fence           4
            [153, 153, 153],  # pole            5
            [250, 170, 30],  # traffic_light    6
            [220, 220, 0],  # traffic_sign     7
            [107, 142, 35],  # vegetation       8
            [152, 251, 152],  # terrain         9
            [70, 130, 180],  # sky               10
            [220, 20, 60],  # person            11
            [255, 0, 0],  # rider             12
            [0, 0, 142],  # car                13
            [0, 0, 70],  # truck               14
            [0, 60, 100],  # bus                15
            [0, 80, 100],  # train              16
            [0, 0, 230],  # motorcycle         17
            [119, 11, 32],  # bicycle           18
        ])

    def zero_out_no_exists_classes(self, classes, semantic_output):
        semantic_output[classes,:,:] = 0
        return semantic_output

    def get_probabilities(self, semantic_output):
        return F.softmax(semantic_output,dim=0)

    def get_class_preds(self, semantic_output):
        _, class_preds = torch.max(semantic_output, axis=0)
        return class_preds

    # SINGLE PRED
    def downsample_prediction(self, class_preds, H, W):
        class_preds = class_preds.unsqueeze(0)
        class_preds = class_preds.unsqueeze(0)
        class_preds_downsampled = F.interpolate(class_preds.type(torch.DoubleTensor), (H, W), mode='nearest')
        class_preds_downsampled = np.array(class_preds_downsampled).astype(int)
        return class_preds_downsampled[0, 0]

    ## MULTIPLE PREDS
    def downsample_predictions(self, class_preds, H, W):
        class_preds = class_preds.unsqueeze(0)
        class_preds_downsampled = F.interpolate(class_preds.type(torch.DoubleTensor), (H, W), mode='nearest')
        class_preds_downsampled = np.array(class_preds_downsampled).astype(int)
        return class_preds_downsampled[0]

    def get_segmented_image(self, class_preds):
        visual = [self.rgbs[p] for p in class_preds]
        visual = np.array(visual)
        return visual

    def get_sky_coords(self, preds):
        # class_preds_downsampled = self.downsample_prediction(class_preds, H=94, W=352)
        indices = np.where(preds == 10)
        coords = list(zip(indices[1], indices[0]))
        coords = np.array(coords)
        return coords



if __name__ == '__main__':

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)



    fig, ax = plt.subplots()
    segmentor = SemanticSegmentor()
    segmentor_helper = SemanticSegmentorHelper()

    im = imread("../../train_data/images/0000005940.png")
    #VERY IMPORTANT! DETECTRON2 MODELS USE BGR INPUT
    im = im[:, :, ::-1]

    outputs = segmentor.segment_image(im)
    #TODO: We know train motorcycle and bicycle do not included in this scene, adjust this dependent to each scene
    outputs = segmentor_helper.zero_out_no_exists_classes(np.array([14,15,16]), outputs)
    # Turn into probabilities

    print(f'hey here: {outputs.shape}')
    outputs = segmentor_helper.get_probabilities(outputs)
    print(f'hey here: {outputs.shape}')


    class_preds = segmentor_helper.get_class_preds(outputs)
    visual = segmentor_helper.get_segmented_image(class_preds)

    plt.imshow(visual)
    plt.show()

    class_preds_downsampled = segmentor_helper.downsample_prediction(class_preds, H=47, W=146)
    visual2 = segmentor_helper.get_segmented_image(class_preds_downsampled)

    plt.imshow(visual2)
    plt.show()