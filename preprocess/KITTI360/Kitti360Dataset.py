#!/usr/bin/python
# -*- coding: utf-8 -*-


#################
## Import modules
#################
# access to OS functionality
import os
# numpy
import numpy
import numpy as np
# open3d
import open3d
# matplotlib for colormaps
import matplotlib.pyplot as plt
from PIL import Image

from preprocess.KITTI360.loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
from preprocess.KITTI360.cameras import CameraPerspective, CameraFisheye
from preprocess.KITTI360.preprocess_utils import latlonToMercator, latToScale, postprocessPoses, convertToHomogeneous


class Kitti360Dataset(object):

    # Constructor
    def __init__(self, seq=0, cam_id=0, kitti360Path='../../data/kitti/KITTI-360'):

        self.sensor_dir = 'velodyne_points'
        self.sick_dir = 'sick_points'

        sequence = '2013_05_28_drive_%04d_sync' % seq
        pose_dir = '%s/data_poses/2013_05_28_drive_%04d_sync/' % (kitti360Path, seq)
        self.pose_file = os.path.join(pose_dir, 'cam%d_to_world.txt' % cam_id)
        self.posestxt_file = os.path.join(pose_dir, 'poses.txt')
        self.oxts_dir = '%s/data_poses_oxts/data_poses/2013_05_28_drive_%04d_sync/oxts/data/' % (kitti360Path, seq)

        self.raw3DPcdPath = os.path.join(kitti360Path, 'data_3d_raw', sequence, self.sensor_dir, 'data')
        self.raw3DPcdTimePath = os.path.join(kitti360Path, 'data_3d_raw', sequence, self.sensor_dir)
        self.raw3DSickPath = os.path.join(kitti360Path, 'data_3d_raw', sequence, self.sick_dir, 'data')
        self.raw3DSickTimePath = os.path.join(kitti360Path, 'data_3d_raw', sequence, self.sick_dir)
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

    '''
    Perspective cameras (image_00, image_01): x = right, y = down, z = forward
    Left fisheye camera (image_02): x = forward, y = down, z = left
    Right fisheye camera (image_03): x = backward, y = down, z = right
    Velodyne: x = forward, y = left, z = up
    GPS/IMU: x = forward, y = right, z = down

    The world coordinate system where we accumulate all 3D points is defined as:
        World coordinate system: x = forward, y = left, z = up



    POSE(from poses.txt): GPS/IMU --> World
    camToPose(calib_cam_to_pose.txt): unrectified camera --> GPS/IMU
    R_rect(perspective.txt): unrectified img --> rectified img
    P_rect_00(perspective.txt): 3x4 intrinsics of the rectified perspective camera
    cam0_to_world(cam0_to_world.txt): rectified camera -->  World
    '''

    # Unrectified perspective camera to GPS
    # 3X4
    def get_unrec_cam_to_gps(self, cam_num=-1):
        if cam_num == -1:
            cam_num = self.camera.cam_id

        TrCamToPose = loadCalibrationCameraToPose(self.fileCameraToPose)
        return TrCamToPose['image_%02d' % cam_num]

    # Unrectified cam0 to velodyne
    def get_unrec_cam0_to_velo(self):
        TrCamToVelo = loadCalibrationRigid(self.fileCameraToPose)
        return TrCamToVelo

    # Sick to velodyne
    def get_sick_to_velo(self):
        TrSickToVelo = loadCalibrationRigid(self.fileSickToVelo)
        return TrSickToVelo

    # GPS/IMU coordinates ---> World Coordinate system
    # 4x4
    def get_gps_to_world(self, frame_no):
        poses = np.loadtxt(self.posestxt_file)
        frames = poses[:, 0].astype(np.int)
        poses = np.reshape(poses[:, 1:], (-1, 3, 4))

        # TODO: dont do this ------>  if pose do not exists get it from oxts
        # while frame_no not in frames:
        #     frame_no -= 1

        frame_index = np.where(frames == frame_no)[0]
        pose = poses[frame_index].squeeze(0)
        pose = convertToHomogeneous(pose)
        return pose

    # Rectified camera coordinates ----> World coordinate system
    def get_rec_cam0_to_world(self, frame_no):
        poses = np.loadtxt(self.pose_file)
        frames = poses[:, 0].astype(np.int)
        poses = np.reshape(poses[:, 1:], (-1, 4, 4))

        # TODO: dont do this ------>  if pose do not exists get it from oxts
        # while frame_no not in frames:
        #     frame_no -= 1

        frame_index = np.where(frames == frame_no)[0]
        pose = poses[frame_index].squeeze(0)
        return pose

    def get_velo_to_rec_cam(self, cam_num=-1):
        if cam_num == -1:
            cam_num = self.camera.cam_id

        # cam_0 to velo
        TrCam0ToVelo = loadCalibrationRigid(self.fileCameraToVelo)

        # all cameras to system center
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
        if cam_num == 0 or cam_num == 1:
            TrVeloToRect = np.matmul(self.camera.R_rect, TrVeloToCam['image_%02d' % cam_num])
        else:
            TrVeloToRect = TrVeloToCam['image_%02d' % cam_num]

        return TrVeloToRect

    def transform_velodyne_to_rec_cam(self, points, cam_id=0):
        TrVeloToRect = self.get_velo_to_rec_cam()
        pointsCam = np.matmul(TrVeloToRect, points.T).T
        pointsCam = pointsCam[:, :3]
        return pointsCam

    def transform_velodyne_points_rec_cam_to_velodyne(self, points, cam_id=0):
        TrVeloToRect = self.get_velo_to_rec_cam()
        TrRectToVelo = np.linalg.inv(TrVeloToRect)
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        points[:, 3] = 1
        points = TrRectToVelo @ points.T
        points = points[:, :3]
        return points

    def project_velodyne_in_rec_cam_to_image_space(self, camera, pointsCam):
        u, v, depth = camera.cam2image(pointsCam.T)
        u = u.astype(int)
        v = v.astype(int)
        return u, v, depth

    def get_mask(self, u, v, depth, camera):
        mask = np.logical_and(np.logical_and(np.logical_and(u >= 0, u < camera.width), v >= 0), v < camera.height)
        # visualize points within 30 meters
        mask = np.logical_and(mask, depth > 0)
        mask = np.logical_and(mask, depth < 30)
        return mask

    def get_depth_map_in_image_space(self, frame):
        camera = self.camera
        points = self.load_velodyne_data(frame)

        points_cam = self.transform_velodyne_to_rec_cam(points, camera.cam_id)
        u, v, depth = self.project_velodyne_in_rec_cam_to_image_space(camera, points_cam)

        # prepare depth map for visualization
        depthMap = np.zeros((camera.height, camera.width))

        # get mask
        mask = self.get_mask(u, v, depth, camera)

        depthMap[v[mask], u[mask]] = depth[mask]

        print(np.max(depthMap))

        return depthMap

    def get_depth_and_coords(self, frame):
        camera = self.camera
        points = self.load_velodyne_data(frame)
        points[:, 3] = 1

        points_cam = self.transform_velodyne_to_rec_cam(points, camera.cam_id)
        u, v, depth = self.project_velodyne_in_rec_cam_to_image_space(camera, points_cam)

        # get mask
        mask = self.get_mask(u, v, depth, camera)

        coords = list(zip(u[mask], v[mask]))
        coords = np.array(coords)
        depth_arr = np.array(depth[mask])

        print(f'before depth shape : {depth_arr.shape}')
        print(coords.shape)

        # TODO: every image has different far/close bounds for now
        # max_depth = np.percentile(depth[mask], 99.5)
        # min_depth = np.percentile(depth[mask], .5)

        max_depth = np.max(depth[mask])
        min_depth = np.min(depth[mask])

        points_world = self.get_points_visible_in_world_coord(frame)

        print(f' points world shape: {points_world.shape}')
        points_world = points_world[:, :3]
        print(depth_arr.shape)
        print(coords.shape)
        print(points_world.shape)
        exit(0)


        save_name = 'points_world_lidar_' + str(frame) + '.npy'
        points_cam = points_cam[mask]
        np.save(save_name, points_world[:, :3])

        pose = self.get_rec_cam0_to_world(frame)
        #
        # line = (points_world - pose[:3, 3])
        # print(pose[:3, 2].reshape(3, 1).shape)
        # print(line.shape)
        # depth = pose[:3, 2].reshape(3, 1).T @ line.T
        # depth_arr = depth.T.squeeze()
        #
        # print(depth_arr.shape)

        # depth_arr = (pose[:3, 2] @ (points_world - pose[:3, 3]).T)
        # print(np.max(depth))
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

        min_depth = np.min(depth_arr)
        max_depth = np.max(depth_arr)

        print(f'after depth shape : {depth_arr.squeeze().shape}')
        print(f'coords shape : {coords.shape}')

        return coords, depth_arr, min_depth, max_depth

    def get_velodyne_points_visible_in_rec_camera(self, frame):
        camera = self.camera
        points = self.load_velodyne_data(frame)
        points[:, 3] = 1

        points_cam = self.transform_velodyne_to_rec_cam(points, camera.cam_id)
        u, v, depth = self.project_velodyne_in_rec_cam_to_image_space(camera, points_cam)

        # get mask
        mask = self.get_mask(u, v, depth, camera)

        # get points visible in the camera by masking
        points_cam = points_cam[mask]
        return points_cam

    def get_points_visible_in_world_coord(self, frame):
        camera = self.camera
        points = self.load_velodyne_data(frame)
        points[:, 3] = 1

        points_cam = self.transform_velodyne_to_rec_cam(points, camera.cam_id)
        u, v, depth = self.project_velodyne_in_rec_cam_to_image_space(camera, points_cam)

        # get mask
        mask = self.get_mask(u, v, depth, camera)

        # get points visible in the rec camera by masking
        points_cam = points_cam[mask]

        points_cam = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1))], axis=1)

        # UNRECTIFY CAMERA
        points_unrec_cam = (np.linalg.inv(self.camera.R_rect) @ points_cam.T).T

        TrCamToPose = self.get_unrec_cam_to_gps()

        ## GPS TO WORLD
        posestxt = self.get_gps_to_world(frame)
        pcd = (posestxt @ TrCamToPose @ points_unrec_cam.T).T
        return pcd

    def get_velodyne_in_world_coord(self, frame):

        pcd = self.load_velodyne_data(frame)

        ## Tr cam pose = calib_cam_to_pose
        ## Tr_cam_velo =  calib_cam_to_velo
        ## Tr pose world = poses.txt
        ## Son_Transform @ .... @ Ilk_Transform @ PTS

        TrVeloToRect = self.get_velo_to_rec_cam()

        pcd = (TrVeloToRect @ pcd.T).T

        # UNRECTIFY CAMERA
        pcd = (np.linalg.inv(self.camera.R_rect) @ pcd.T).T

        TrCamToGPS = self.get_unrec_cam_to_gps()

        # GPS TO WORLD
        posestxt = self.get_gps_to_world(frame)

        pcd = (posestxt @ TrCamToGPS @ pcd.T).T

        return pcd

    def create_poses_bounds_and_gt_depths(self, frames):

        height, width, focal = self.camera.height, self.camera.width, self.camera.focal
        hwf = np.array([height, width, focal]).reshape(3, 1)

        poses = []
        min_max_depths = []
        depth_data_list = []
        for frame in frames:
            pose = self.get_rec_cam0_to_world(frame)
            poses.append(pose)
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

    def visualize_data(self, frame, cam_id, vis2d=True):
        depth_map = self.get_depth_map_in_image_space(frame)

        layout = (2, 1) if cam_id in [0, 1] else (1, 2)
        fig, axs = plt.subplots(*layout, figsize=(18, 12))

        image_path = self.get_image_path(frame)

        # color map for visualizing depth map
        cm = plt.get_cmap('jet')

        colorImage = np.array(Image.open(image_path)) / 255.
        depthImage = cm(depth_map / depth_map.max())[..., :3]
        colorImage[depth_map > 0] = depthImage[depth_map > 0]

        axs[0].imshow(depth_map, cmap='jet')
        axs[0].title.set_text('Projected Depth')
        axs[0].axis('off')
        axs[1].imshow(colorImage)
        axs[1].title.set_text('Projected Depth Overlaid on Image')
        axs[1].axis('off')
        plt.suptitle('Sequence %04d, Camera %02d, Frame %010d' % (seq, cam_id, frame))
        plt.show()

    def convertOxtsToPose(self, frame):
        ''' converts a list of oxts measurements into metric poses,
        starting at (0,0,0) meters, OXTS coordinates are defined as
        x = forward, y = right, z = down (see OXTS RT3000 user manual)
        afterwards, pose{i} contains the transformation which takes a
        3D point in the i'th frame and projects it into the oxts
        coordinates of the first frame. '''

        oxts_file = os.path.join(self.oxts_dir, '%010d.txt' % frame)
        oxts = np.loadtxt(oxts_file)

        single_value = not isinstance(oxts, list)
        if single_value:
            oxts = [oxts]

        # origin in OXTS coordinate
        origin_oxts = [48.9843445, 8.4295857]  # lake in Karlsruhe

        # compute scale from lat value of the origin
        scale = latToScale(origin_oxts[0])

        # origin in Mercator coordinate
        ox, oy = latlonToMercator(origin_oxts[0], origin_oxts[1], scale)
        origin = np.array([ox, oy, 0])

        pose = []

        # for all oxts packets do
        for i in range(len(oxts)):

            # if there is no data => no pose
            if not len(oxts[i]):
                pose.append([])
                continue

            # translation vector
            tx, ty = latlonToMercator(oxts[i][0], oxts[i][1], scale)
            t = np.array([tx, ty, oxts[i][2]])

            # rotation matrix (OXTS RT3000 user manual, page 71/92)
            rx = oxts[i][3]  # roll
            ry = oxts[i][4]  # pitch
            rz = oxts[i][5]  # heading
            Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)],
                           [0, np.sin(rx), np.cos(rx)]])  # base => nav  (level oxts => rotated oxts)
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0],
                           [-np.sin(ry), 0, np.cos(ry)]])  # base => nav  (level oxts => rotated oxts)
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0],
                           [0, 0, 1]])  # base => nav  (level oxts => rotated oxts)
            R = np.matmul(np.matmul(Rz, Ry), Rx)

            # normalize translation
            t = t - origin

            # add pose
            pose.append(np.vstack((np.hstack((R, t.reshape(3, 1))), np.array([0, 0, 0, 1]))))

        if single_value:
            pose = pose[0]

        pose = np.array(pose)

        # convert coordinate system from
        #   x=forward, y=right, z=down
        # to
        #   x=down, y=right, z=backwards
        pose = postprocessPoses(pose)

        pose = np.array(pose)

        return pose

if __name__ == '__main__':
    visualizeIn2D = True
    # sequence index
    seq = 0
    # set it to 0 or 1 for projection to perspective images
    #           2 or 3 for projecting to fisheye images
    cam_id = 0
    frame = 5930

    dataset = Kitti360Dataset(seq, cam_id)

    # depth_arr = dataset.get_depth_map(5930)

    # dataset.visualize_sick_frames()

    # dataset.get_unrec_cam_to_gps()

    # plt.imshow(depth_arr)
    # plt.show()
    #dataset.visualize_data(frame, cam_id, visualizeIn2D)
    out = dataset.dense_map(frame, grid=1)
    plt.figure(figsize=(20, 40))
    plt.imsave("depth_map_%06d.png" % frame, out)
    #pcd = dataset.get_accumulated_pointcloud()
    #dataset.convert_accumulated_pcd_to_cam(pcd, frame)

    #dataset.calculate_world_to_cam0_accumulated_pcd()

    #dataset.load_sick_data()

    #dataset.visualize_pcd(dataset.get_points_visible_in_camera())



