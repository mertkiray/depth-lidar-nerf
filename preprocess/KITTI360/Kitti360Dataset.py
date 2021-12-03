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

from loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
from cameras import CameraPerspective, CameraFisheye
from preprocess.KITTI360.utils import latlonToMercator, latToScale, postprocessPoses


class Kitti360Dataset(object):

    # Constructor
    def __init__(self, seq=0, cam_id=0, kitti360Path='../../data/kitti/KITTI-360'):

        self.sensor_dir = 'velodyne_points'

        sequence = '2013_05_28_drive_%04d_sync' % seq
        pose_dir = '%s/data_poses/2013_05_28_drive_%04d_sync/' % (kitti360Path, seq)
        self.pose_file = os.path.join(pose_dir, 'cam%d_to_world.txt' % cam_id)
        self.posestxt_file = os.path.join(pose_dir, 'poses.txt')
        self.oxts_dir = '%s/data_poses_oxts/data_poses/2013_05_28_drive_%04d_sync/oxts/data/' % (kitti360Path, seq)

        self.raw3DPcdPath = os.path.join(kitti360Path, 'data_3d_raw', sequence, self.sensor_dir, 'data')
        self.raw2DImagePath = os.path.join(kitti360Path, 'data_2d_raw', sequence, ('image_%02d' % cam_id), 'data_rect')
        self.fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
        self.fileCameraToPose = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_pose.txt')

        if cam_id in [0, 1]:
            self.camera = CameraPerspective(kitti360Path, sequence, cam_id)
        else:
            raise RuntimeError('Unknown camera ID!')

    def get_image_path(self, frame):
        image_file = os.path.join(self.raw2DImagePath, '%010d.png' % frame)
        return image_file


    def get_transform(self, cam_id=0):
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
        if cam_id == 0 or cam_id == 1:
            TrVeloToRect = np.matmul(self.camera.R_rect, TrVeloToCam['image_%02d' % cam_id])
        else:
            TrVeloToRect = TrVeloToCam['image_%02d' % cam_id]

        return TrVeloToRect


    def get_velodyne_points_in_camera_coord(self, points, cam_id=0):
        TrVeloToRect = self.get_transform(cam_id)
        pointsCam = np.matmul(TrVeloToRect, points.T).T
        pointsCam = pointsCam[:, :3]
        return pointsCam


    def transform_velodyne_points_cam_to_velo(self, points, cam_id=0):
        TrVeloToRect = self.get_transform(cam_id)
        TrRectToVelo = np.linalg.inv(TrVeloToRect)



        points = np.concatenate([points,np.ones((points.shape[0], 1))], axis=1)
        points[:, 3] = 1

        points = TrRectToVelo @ points.T
        points = points[:, :3]
        return points

    def project_velodyne_to_image_space(self, camera, pointsCam):
        u, v, depth = camera.cam2image(pointsCam.T)
        u = u.astype(int)
        v = v.astype(int)
        return u, v, depth

    def get_depth_map(self, frame):
        camera = self.camera
        points = self.load_velodyne_data(frame)

        points_cam = self.get_velodyne_points_in_camera_coord(points, camera.cam_id)
        u, v, depth = self.project_velodyne_to_image_space(camera, points_cam)

        # prepare depth map for visualization
        depthMap = np.zeros((camera.height, camera.width))

        #get mask
        mask = self.get_mask(u, v, depth, camera)

        depthMap[v[mask], u[mask]] = depth[mask]

        return depthMap


    def get_depth_and_coords(self, frame):
        camera = self.camera
        points = self.load_velodyne_data(frame)
        points[:, 3] = 1

        points_cam = self.get_velodyne_points_in_camera_coord(points, camera.cam_id)
        u, v, depth = self.project_velodyne_to_image_space(camera, points_cam)

        #get mask
        mask = self.get_mask(u, v, depth, camera)

        coords = list(zip(u[mask], v[mask]))
        coords = np.array(coords)
        depth_arr = np.array(depth[mask])

        # TODO: every image has different far/close bounds for now
        max_depth = np.percentile(depth[mask], 99.9)
        min_depth = np.percentile(depth[mask], 0.1)

        return coords, depth_arr, min_depth, max_depth

    def get_points_visible_in_camera(self, frame):
        camera = self.camera
        points = self.load_velodyne_data(frame)
        points[:, 3] = 1

        points_cam = self.get_velodyne_points_in_camera_coord(points, camera.cam_id)
        u, v, depth = self.project_velodyne_to_image_space(camera, points_cam)

        # get mask
        mask = self.get_mask(u, v, depth, camera)

        # get points visible in the camera by masking
        points_cam = points_cam[mask]
        return points_cam

    def get_points_visible_in_world_coord(self, frame):
        camera = self.camera
        points = self.load_velodyne_data(frame)
        points[:, 3] = 1

        points_cam = self.get_velodyne_points_in_camera_coord(points, camera.cam_id)
        u, v, depth = self.project_velodyne_to_image_space(camera, points_cam)

        # get mask
        mask = self.get_mask(u, v, depth, camera)

        # get points visible in the camera by masking
        points_cam = points_cam[mask]

        points_cam = np.concatenate([points_cam, np.ones((points_cam.shape[0], 1))], axis=1)

        TrCamToPose = self.camera.camToPose

        posestxt = self.get_posestxt_of_frame(frame)

        pcd = (posestxt @ TrCamToPose @ points_cam.T).T

        return pcd

    def get_mask(self, u, v, depth, camera):
        mask = np.logical_and(np.logical_and(np.logical_and(u >= 0, u < camera.width), v >= 0), v < camera.height)
        # visualize points within 30 meters
        mask = np.logical_and(mask, depth > 0)
        return mask

    def load_velodyne_data(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd, [-1, 4])
        pcd[:, 3] = 1
        return pcd

    def load_sick_data(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd, [-1, 2])
        pcd = np.concatenate([np.zeros_like(pcd[:, 0:1]), -pcd[:, 0:1], pcd[:, 1:2]], axis=1)
        return pcd

    def visualize_pcd(self, points):
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points[:, :3])
        open3d.visualization.draw_geometries([pcd])


    def get_velodyne_in_world_coord(self, frame):

        pcd = self.load_velodyne_data(frame)

        ## Tr cam pose = calib_cam_to_pose
        ## Tr_cam_velo =  calib_cam_to_velo
        ## Tr pose world = poses.txt
        ## Son_Transform @ .... @ Ilk_Transform @ PTS

        TrCam0ToVelo = loadCalibrationRigid(self.fileCameraToVelo)
        TrVeloToCam = np.linalg.inv(TrCam0ToVelo)

        TrCamToPose = self.camera.camToPose

        posestxt = self.get_posestxt_of_frame(frame)


        pcd = (posestxt @ TrCamToPose @ TrVeloToCam @ pcd.T).T

        return pcd


    def visualize_pcd_and_poses(self, frames):

        TrCam0ToVelo = loadCalibrationRigid(self.fileCameraToVelo)
        TrVeloToCam = np.linalg.inv(TrCam0ToVelo)

        TrCamToPose = self.camera.camToPose


        pcds = []
        poses = []


        for frame in frames:
            # velodyne data
            pcd = self.load_velodyne_data(frame)

            # velodyne to camera
            pcd = np.matmul(pcd, TrVeloToCam)
            # camera to pose
            pcd = np.matmul(pcd, TrCamToPose)
            pcds.append(pcd)

            pose = self.get_pose_of_frame(frame)
            poses.append(pose)

        pcds = np.array(pcds)
        poses = np.array(poses).squeeze(1)

        pcds = pcds[0]
        pcds = pcds[:,:3]

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pcds)



        pose = poses[:,:3, 3]


        pcd_pose = open3d.geometry.PointCloud()
        pcd_pose.points = open3d.utility.Vector3dVector(pose)

        pcd_pose.paint_uniform_color([1, 0.706, 0])

        open3d.visualization.draw_geometries([pcd, pcd_pose])

    def get_pose_of_frame(self, frame_no):

        poses = np.loadtxt(self.pose_file)
        frames = poses[:, 0].astype(np.int)
        poses = np.reshape(poses[:, 1:], (-1, 4, 4))

        while frame_no not in frames:
            frame_no -= 1

        frame_index = np.where(frames == frame_no)[0]

        return poses[frame_index].squeeze(0)


    def get_posestxt_of_frame(self, frame_no):

        poses = np.loadtxt(self.posestxt_file)
        frames = poses[:, 0].astype(np.int)
        poses = np.reshape(poses[:, 1:], (-1, 3, 4))

        while frame_no not in frames:
            frame_no -= 1

        frame_index = np.where(frames == frame_no)[0]
        pose = poses[frame_index].squeeze(0)
        pose = np.concatenate((pose, np.array([0., 0., 0., 1.]).reshape(1, 4)))

        return pose


    def create_poses_bounds_and_gt_depths(self, frames):

        height, width, focal = self.camera.height, self.camera.width, self.camera.focal
        hwf = np.array([height, width, focal]).reshape(3, 1)

        poses = []
        min_max_depths = []
        depth_data_list = []
        for frame in frames:
            pose = self.get_pose_of_frame(frame)
            #pose = np.matmul(np.matmul(pose, self.camera.camToPose), np.linalg.inv(self.camera.R_rect))
            poses.append(pose)
            coord, depth, min_depth, max_depth = self.get_depth_and_coords(frame)

            depth_data_list.append({'depth': np.array(depth), 'coord': np.array(coord), 'weight': np.ones(depth.shape)})
            min_max_depths.append([min_depth, max_depth])

        min_max_depths = np.array(min_max_depths)
        poses = np.array(poses)[:,:3,:]
        #poses = np.array(poses)[:,:3,:]
        hwf = np.broadcast_to(hwf, (poses.shape[0], hwf.shape[0], hwf.shape[1]))
        poses = np.append(poses, hwf, axis=2)
        poses = poses.reshape(poses.shape[0], poses.shape[1] * poses.shape[2])


        poses = np.concatenate([poses, min_max_depths], axis=1)

        numpy.save('../../train_data/poses_bounds.npy', poses)

        # dict: 'depth': np.array(depth), 'coord': np.array(coord), 'weight': np.ones(depth.shape)
        numpy.save('../../train_data/depth_gt.npy', depth_data_list)

        return


    def visualize_data(self, frame, cam_id, vis2d=True):

        if vis2d:

            depth_map = self.get_depth_map(frame)

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

        else:
            # visualize raw 3D scans in 3D
            points = self.load_velodyne_data(frame)
            self.visualize_pcd(points)

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
        #   x=down, y=right, z=backwardss
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
    depth_arr = dataset.get_depth_map(5930)

    plt.imshow(depth_arr)
    plt.show()

    dataset.visualize_data(frame, cam_id, visualizeIn2D)
    # dataset.visualize_pcd(dataset.get_points_visible_in_camera())



