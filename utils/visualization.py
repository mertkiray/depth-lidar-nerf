import numpy as np
import os
import imageio
from pathlib import Path

from matplotlib import pyplot as plt

from .read_write_model import *


def visualize_keypoints(basedir, factor=8, bd_factor=.75):
    imgdir = imgdir = os.path.join(basedir, 'images')
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    imgs = [imageio.imread(f)[...,:3] for f in imgfiles]
    

    data_file = Path(basedir) / 'colmap_depth.npy'
    
    images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
    points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')

    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)

    data_list = []
    for id_im in range(1, len(images)+1):
        depth_list = []
        coord_list = []
        weight_list = []
        # for i in range(len(images[id_im].xys)):
        #     point2D = images[id_im].xys[i]
        #     id_3D = images[id_im].point3D_ids[i]
        #     if id_3D == -1:
        #         continue
        #     point3D = points[id_3D].xyz
        #     depth = (poses[id_im-1,:3,2].T @ (point3D - poses[id_im-1,:3,3])) * sc
        #     if depth < bds_raw[id_im-1,0] * sc or depth > bds_raw[id_im-1,1] * sc:
        #         continue
        #     err = points[id_3D].error
        #     weight = 2 * np.exp(-(err/Err_mean)**2)
        #     depth_list.append(depth)
        #     coord_list.append(point2D/factor)
        #     weight_list.append(weight)
        # print(id_im, np.min(depth_list), np.max(depth_list), np.mean(depth_list))
        # data_list.append({"depth":np.array(depth_list), "coord":np.array(coord_list), "weight":np.array(weight_list)})
        imageio.imwrite(os.path.join(basedir,'visual', '{:03d}_visual.png'.format(id_im-1)), imgs[id_im-1])
   
    return 

def visualize_depths_as_image(depth):
    cm = plt.get_cmap('jet')
    depthImage = cm(depth / depth.max())[..., :3]
    depthImageWorld = cm((1 / (1 - depth)) / (1 / (1 - depth)).max())[..., :3]

    return depthImage, depthImageWorld

def visualize_depths_masked_uv(depth, coord_array):

    depth_map = np.zeros((depth.shape[0], depth.shape[1]))

    for i, coord in enumerate(coord_array):
        depth_map[int(coord[1]), int(coord[0])] = depth[int(coord[1]), int(coord[0])]

    depth_map = np.ma.masked_where(depth_map == 0, depth_map)

    cm = plt.get_cmap('jet')
    cm.set_bad(color='black')
    depthImage = cm(depth_map / depth_map.max())[..., :3]

    return depthImage



def visualize_depths_on_image(depth, image):

    print(image.shape)

    coord_array = np.array(depth['coord'])
    depth_array = np.array(depth['depth'])
    print(coord_array.shape)
    depth_map = np.zeros((image.shape[0], image.shape[1]))

    print(f'depth_map shape: {depth_map.shape}')

    for i, coord in enumerate(coord_array):
        depth_map[int(coord[1]), int(coord[0])] = depth_array[i]

    depth_map = np.ma.masked_where(depth_map == 0, depth_map)

    # color map for visualizing depth map
    cm = plt.get_cmap('jet')
    cm.set_bad(color='black')

    colorImage = np.array(image)

    print(f'color image shape : {colorImage.shape}')
    depthImage = cm(depth_map / depth_map.max())[..., :3]
    colorImage[depth_map > 0] = depthImage[depth_map > 0]

    return depthImage, colorImage

def main():
    visualize_keypoints("/data2/kangled/datasets/DSNeRF/dtu/scan21_all")

if __name__ == "__main__":
    main()