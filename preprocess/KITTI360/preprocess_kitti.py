import os
import numpy as np
import yaml
from Kitti360Dataset import Kitti360Dataset


def preprocess_kitti():
    with open('config.yaml', 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    cam_id = config['cam_id']
    seq_id = config['seq_id']
    image_folder = config['image_folder']

    files = os.listdir(image_folder)
    image_nums = []
    for file in files:
        file = file.split('.png')
        image_nums.append(int(file[0].lstrip('0')))

    dataset = Kitti360Dataset(seq_id, cam_id)

    processed_data = []

    dataset.create_poses_bounds_and_gt_depths(image_nums)

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
    arr = np.load('colmap_depth.npy', allow_pickle=True)
    print(arr[0]['depth'].max())

if __name__ == '__main__':
    preprocess_kitti()
