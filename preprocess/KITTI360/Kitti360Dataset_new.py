#!/usr/bin/python
# -*- coding: utf-8 -*-


#################
## Import modules
#################
# access to OS functionality
import os
# numpy
import imageio
import numpy
import numpy as np
# open3d
import open3d
# matplotlib for colormaps
import matplotlib.pyplot as plt
from PIL import Image

from loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
from cameras import CameraPerspective, CameraFisheye
from preprocess_utils import latlonToMercator, latToScale, postprocessPoses, convertToHomogeneous
import utils.depth_map_utils as depth_utils

class Kitti360DatasetNew(object):

    # Constructor
    def __init__(self, seq=0, cam_id=0, kitti360Path='../../data/kitti/KITTI-360'):

        self.sensor_dir = 'velodyne_points'
        self.sick_dir = 'sick_points'

        sequence = '2013_05_28_drive_%04d_sync' % seq
        pose_dir = '%s/data_poses/2013_05_28_drive_%04d_sync/' % (kitti360Path, seq)
        self.cam0_to_world = os.path.join(pose_dir, 'cam%d_to_world.txt' % cam_id)
        self.poses = os.path.join(pose_dir, 'poses.txt')
        self.oxts = '%s/data_poses_oxts/data_poses/2013_05_28_drive_%04d_sync/oxts/data/' % (kitti360Path, seq)

        self.raw3DPcdPath = os.path.join(kitti360Path, 'data_3d_raw', sequence, self.sensor_dir, 'data')
        self.raw3DSickPath = os.path.join(kitti360Path, 'data_3d_raw', sequence, self.sick_dir, 'data')
        self.raw2DImagePath = os.path.join(kitti360Path, 'data_2d_raw', sequence, ('image_%02d' % cam_id), 'data_rect')

        self.fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
        self.fileCameraToPose = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_pose.txt')
        self.fileSickToVelo = os.path.join(kitti360Path, 'calibration', 'calib_sick_to_velo.txt')




        if cam_id in [0, 1]:
            self.camera = CameraPerspective(kitti360Path, sequence, cam_id)
        else:
            raise RuntimeError('Unknown camera ID!')

    def get_image_path(self, frame):
        image_file = os.path.join(self.raw2DImagePath, '%010d.png' % frame)
        return image_file

    def load_velodyne_data(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd, [-1, 4])
        pcd[:, 3] = 1
        return pcd

    # Rectified camera coordinates ----> World coordinate system
    def cam2world(self, frame):
        poses = np.loadtxt(self.cam0_to_world)
        frames = poses[:, 0].astype(int)
        poses = np.reshape(poses[:, 1:], (-1, 4, 4))

        # TODO: dont do this ------>  if pose do not exists get it from oxts
        # while frame_no not in frames:
        #     frame_no -= 1

        frame_index = np.where(frames == frame)[0]
        pose = poses[frame_index].squeeze(0)
        return pose

        # GPS/IMU coordinates ---> World Coordinate system
        # 4x4

    def pose(self, frame):
        poses = np.loadtxt(self.poses)
        frames = poses[:, 0].astype(int)
        poses = np.reshape(poses[:, 1:], (-1, 3, 4))

        # TODO: dont do this ------>  if pose do not exists get it from oxts
        # while frame_no not in frames:
        #     frame_no -= 1

        frame_index = np.where(frames == frame)[0]
        pose = poses[frame_index].squeeze(0)
        pose = convertToHomogeneous(pose)
        return pose

    def create_poses_bounds_and_gt_depths(self, frames, sky_coords=None):
        height, width, focal = self.camera.height, self.camera.width, self.camera.focal
        hwf = np.array([height, width, focal]).reshape(3, 1)

        poses = []
        min_max_depths = []
        depth_data_list = []
        for i, frame in enumerate(frames):
            pose = self.cam2world(frame)
            poses.append(pose)
            if sky_coords is not None:
                coord, depth, min_depth, max_depth = self.get_depth_and_coords(frame, sky_coords=sky_coords[i])
            else:
                coord, depth, min_depth, max_depth = self.get_depth_and_coords(frame)

            depth_data_list.append({'depth': np.array(depth), 'coord': np.array(coord), 'weight': np.ones(depth.shape)})
            min_max_depths.append([min_depth, max_depth])

        min_max_depths = np.array(min_max_depths)
        poses = np.array(poses)[:, :3, :]
        hwf = np.broadcast_to(hwf, (poses.shape[0], hwf.shape[0], hwf.shape[1]))
        poses = np.append(poses, hwf, axis=2)
        poses = poses.reshape(poses.shape[0], poses.shape[1] * poses.shape[2])

        poses = np.concatenate([poses, min_max_depths], axis=1)
        numpy.save('../../train_data/poses_bounds.npy', poses)
        # dict: 'depth': np.array(depth), 'coord': np.array(coord), 'weight': np.ones(depth.shape)
        numpy.save('../../train_data/depth_gt.npy', depth_data_list)
        return


    def get_depth_and_coords(self, frame, sky_coords=None):

        cam_id = self.camera.cam_id

        TrCam0ToVelo = loadCalibrationRigid(self.fileCameraToVelo)
        TrCamToPose = loadCalibrationCameraToPose(self.fileCameraToPose)

        # velodyne to all cameras
        TrVeloToCam = {}
        for k, v in TrCamToPose.items():
            # Tr(cam_k -> velo) = Tr(cam_k -> cam_0) @ Tr(cam_0 -> velo)
            TrCamkToCam0 = np.linalg.inv(TrCamToPose['image_00']) @ TrCamToPose[k]
            TrCamToVelo = TrCam0ToVelo @ TrCamkToCam0
            # Tr(velo -> cam_k)
            TrVeloToCam[k] = np.linalg.inv(TrCamToVelo)

        # take the rectification into account for perspective cameras
        if cam_id == 0 or cam_id == 1:
            TrVeloToRect = np.matmul(self.camera.R_rect, TrVeloToCam['image_%02d' % cam_id])
        else:
            TrVeloToRect = TrVeloToCam['image_%02d' % cam_id]

        pose = self.pose(frame)

        ## LIDAR PCD IN VELODYNE COORDINATES
        lidar_pcd = self.load_velodyne_data(frame)

        # transfrom velodyne points to camera coordinate
        pointsCamStart = np.matmul(TrVeloToRect, lidar_pcd.T).T
        pointsCam = pointsCamStart[:, :3]

        # project to image space
        u, v, depth, points_projected = self.camera.cam2imageNew(pointsCam.T)
        u = u.astype(int)
        v = v.astype(int)




        mask = np.logical_and(np.logical_and(np.logical_and(u >= 0, u < self.camera.width), v >= 0), v < self.camera.height)
        # visualize points within 30 meters
        mask = np.logical_and(mask, depth > 0)
        #mask = np.logical_and(mask, depth<30)
        #mask = np.logical_and(mask, depth<99)


        coords = list(zip(u[mask], v[mask]))
        coords = np.array(coords)

        # print(f'TrCam0ToVelo shape: {TrCam0ToVelo.shape}')
        # print(f"TrCamToPose shape: {TrCamToPose['image_%02d' % cam_id].shape}")
        # print(f'Pose shape: {pose.shape}')

        depth_arr = depth[mask]

        # min_depth = np.min(depth_arr)
        # max_depth = np.max(depth_arr)

        min_depth = np.percentile(depth_arr, .1)
        max_depth = np.percentile(depth_arr, 99.9)

        # if sky_coords is not None:
        #     coords = np.concatenate([coords, sky_coords])
        #     max_depth_sky = np.ones(shape=(sky_coords.shape[0],)) * (max_depth)
        #     depth_arr = np.concatenate([depth_arr, max_depth_sky])

        depth_arr, coords = self.complete_depth(depth_arr, coords, 9999999, sky_coords)
        min_depth = np.percentile(depth_arr, .1)
        max_depth = np.percentile(depth_arr, 99.9)

        return coords, depth_arr, min_depth, max_depth

    def complete_depth(self, depth_arr, coords, max_depth, sky_coords):
        cm = plt.get_cmap('jet')
        cm.set_bad(color='white')

        width, heigth = self.camera.width, self.camera.height
        depth_image = np.zeros(shape=(heigth, width))

        depth_image[coords[...,1], coords[..., 0]] = depth_arr
        #depth_image = np.ma.masked_where(depth_image == 0, depth_image)
        #print(depth_image[0,0])
        #depth_image = cm(depth_image / depth_image.max())[..., :3]
        # plt.imshow(depth_image)
        # plt.show()


        mod_depth_arr = depth_utils.fill_in_multiscale(depth_image, max_depth=depth_arr.max()+1,
                                                                     extrapolate=True)
        mod_depth_arr[sky_coords[...,1], sky_coords[..., 0]] = max_depth

        # depth_image = np.ma.masked_where(mod_depth_arr == 0, mod_depth_arr)
        # depth_image = cm(depth_image / depth_image.max())[..., :3]
        # plt.imshow(depth_image)
        # plt.show()
        # mod_depth_arr = depth_utils.fill_in_multiscale(mod_depth_arr, max_depth=depth_arr.max()+1,
        #                                                extrapolate=False)
        # depth_image = np.ma.masked_where(mod_depth_arr == 0, mod_depth_arr)
        # depth_image = cm(depth_image / depth_image.max())[..., :3]
        # plt.imshow(depth_image)
        # plt.show()

        depth_arr = mod_depth_arr.flatten()
        y, x = np.indices(mod_depth_arr.shape).reshape(-1, len(depth_arr))
        coords = list(zip(x, y))

        depth_arr_end = []
        coords_end = []
        for i, depth in enumerate(depth_arr):
            if depth != 0:
                depth_arr_end.append(depth)
                coords_end.append(coords[i])
        depth_arr_end = np.array(depth_arr_end)

        # depth_image = np.ma.masked_where(mod_depth_arr == 0, mod_depth_arr)
        # depth_image = cm(depth_image / depth_image.max())[..., :3]
        # plt.imshow(depth_image)
        # plt.show()
        #
        # exit(0)
        return depth_arr_end, coords_end


    def trash(self):
        ## VELO TO CAM0
        lidar_pcd = (np.linalg.inv(TrCam0ToVelo) @ lidar_pcd.T).T
        print(lidar_pcd.shape)


        # lidar_pcd = (np.linalg.inv(self.camera.R_rect) @ lidar_pcd.T).T
        # print(self.camera.R_rect.shape)

        ## LIDAR CAM0 TO POSE
        lidar_pcd = (TrCamToPose['image_%02d' % cam_id] @ lidar_pcd.T).T
        print(lidar_pcd.shape)

        ## LIDAR POSE TO WORLD
        lidar_pcd = (pose @ lidar_pcd.T).T
        print(lidar_pcd.shape)

        lidar_pcd = lidar_pcd[mask]
        lidar_pcd = lidar_pcd[:, :3]
        print('==============')
        print(lidar_pcd.shape)

        save_name = 'points_world_lidar_' + str(frame) + '.npy'
        np.save(save_name, lidar_pcd[:, :3])



        cam2world = self.cam2world(frame)

        print(cam2world[:3, 2].T.shape)
        print(cam2world[:3, 3].shape)

        print('XXXXXXXXXXXXXXX')
        print((lidar_pcd - cam2world[:3, 3]).shape)
        print(cam2world[:3, 2].T.shape)
        print(depth[mask].shape)
        print('XXXXXXXXXXXXXXX')


        depth_arr = (cam2world[:3, 2].T @ (lidar_pcd - cam2world[:3, 3]).T)
        print(depth_arr[0])
        depth_arr = depth[mask]
        print(depth_arr[0])


        cam_xyz = cam2world[:,3]
        #depth_arr = np.sqrt(np.square(cam_xyz[0] - lidar_pcd[...,0]) + np.square(cam_xyz[1] - lidar_pcd[...,1]) + np.square(cam_xyz[2] - lidar_pcd[...,2]))

        print(depth_arr.shape)



    def dense_map(self, Pts, n, m, grid):
        ng = 2 * grid + 1

        mX = np.zeros((m, n)) + float("inf")
        mY = np.zeros((m, n)) + float("inf")
        mD = np.zeros((m, n))
        mX[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
        mY[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
        mD[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[2]

        KmX = np.zeros((ng, ng, m - ng, n - ng))
        KmY = np.zeros((ng, ng, m - ng, n - ng))
        KmD = np.zeros((ng, ng, m - ng, n - ng))

        for i in range(ng):
            for j in range(ng):
                KmX[i, j] = mX[i: (m - ng + i), j: (n - ng + j)] + i
                KmY[i, j] = mY[i: (m - ng + i), j: (n - ng + j)] + i
                KmD[i, j] = mD[i: (m - ng + i), j: (n - ng + j)]
        S = np.zeros_like(KmD[0, 0])
        Y = np.zeros_like(KmD[0, 0])

        for i in range(ng):
            for j in range(ng):
                s = 1 / np.sqrt(KmX[i, j] * KmX[i, j] + KmY[i, j] * KmY[i, j])
                Y = Y + s * KmD[i, j]
                S = S + s

        S[S == 0] = 1
        out = np.ones((m, n))
        out = np.negative(out)

        out[grid + 1: -grid, grid + 1: -grid] = Y / S
        return out

if __name__ == '__main__':
    seq = 0
    cam_id = 0
    frame = 5930


    dataset = Kitti360DatasetNew(seq, cam_id)

    dataset.get_depth_and_coords(frame)