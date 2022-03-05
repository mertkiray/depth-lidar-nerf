import os

import imageio
import numpy as np
import yaml
from matplotlib import pyplot as plt

from Kitti360Dataset_new import Kitti360DatasetNew
from segmentor import SemanticSegmentor, SemanticSegmentorHelper


def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)

def preprocess_kitti():
    with open('config.yaml', 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    cam_id = config['cam_id']
    seq_id = config['seq_id']
    image_folder = config['image_folder']
    segmentor_config = config['segmentor_config']
    segmentor_weights = config['segmentor_weights']

    files = sorted(os.listdir(image_folder))
    image_nums = []
    image_paths = []
    for file in files:
        file_path = image_folder + '/' + file
        image_paths.append(file_path)
        file = file.split('.png')
        image_nums.append(int(file[0].lstrip('0')))

    image_nums.sort()


    print(image_paths)
    segmentor = SemanticSegmentor(segmentor_config, segmentor_weights)
    segmentor_helper = SemanticSegmentorHelper()

    segmentation_gt = []
    sky_coords = []
    for image in image_paths:
        im = imread(image)
        #VERY IMPORTANT! DETECTRON2 MODELS USE BGR INPUT
        im = im[:, :, ::-1]
        outputs = segmentor.segment_image(im)
        # TODO: We know train motorcycle and bicycle do not included in this scene, adjust this dependent to each scene
        outputs = segmentor_helper.zero_out_no_exists_classes(np.array([11,12, 14, 15, 16]), outputs)
        # Turn into probabilities
        outputs = segmentor_helper.get_probabilities(outputs)

        class_preds = segmentor_helper.get_class_preds(outputs)
        segmentation_gt.append(class_preds)

        sky_coord = segmentor_helper.get_sky_coords(class_preds)
        sky_coords.append(sky_coord)

        # visual = segmentor_helper.get_segmented_image(class_preds)
        #
        # plt.imshow(visual)
        # plt.show()

    segmentation_gt = np.stack(segmentation_gt, axis=0)

    sky_coords = np.array(sky_coords)

    segmentation_result = {'segmentations': segmentation_gt, 'num_classes': 19}
    np.save('../../train_data/segmentation_gt.npy', segmentation_result)

    dataset = Kitti360DatasetNew(seq_id, cam_id)
    dataset.create_poses_bounds_and_gt_depths(image_nums, sky_coords)

    # for frame in config['frame_no']:
    #
    #     image_path = dataset.get_image_path(frame,cam_id)
    #     depth_map = dataset.get_depth_map(frame, cam_id)
    #     print(depth_map.shape)
    #
    #     data = {'image': image_path, 'depth_map': depth_map}
    #     processed_data.append(data)
    #
    # np.save('kitti_360.npy', processed_data, allow_pickle=True)


def read_data():
    arr = np.load('../../train_data/segmentation_gt.npy', allow_pickle=True)
    arr = arr.item()
    print(arr['segmentations'])
    print(arr['num_classes'])

if __name__ == '__main__':
    preprocess_kitti()
