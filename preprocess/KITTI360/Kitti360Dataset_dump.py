
# returns 4x4 pcd
def get_accumulated_pointcloud(self):
    pcdFile = os.path.join('../../../accumulated_kitti/2013_05_28_drive_0000_sync_005900_006000/lidar_points_all.dat')
    if not os.path.isfile(pcdFile):
        raise RuntimeError('%s does not exist!' % pcdFile)
    pcd = np.loadtxt(pcdFile, dtype=np.float32)
    pcd = pcd[:, :4]
    pcd[: ,3] = 1
    print(f'Point cloud shape: {pcd.shape}')



    # gps_pose_file = os.path.join('../../../accumulated_kitti/2013_05_28_drive_0000_sync_005900_006000/lidar_pose.dat')
    # gps_poses = np.loadtxt(gps_pose_file)
    # frames = gps_poses[:, 0].astype(np.int)
    # poses_gps = np.reshape(gps_poses[:, 1:], (-1, 4, 4))
    #
    #
    # poses_cam0 = []
    # for i in range(21):
    #     i = 5930 + i
    #     print(i)
    #     pose = self.get_rec_cam0_to_world(i)
    #     poses_cam0.append(pose)
    #
    #
    #
    # poses_cam0 = np.array(poses_cam0)
    # print(f'poses shape: {poses_cam0.shape}')
    #
    #
    # poses_gps_xyz = poses_gps[:, :3, 3]
    # poses_cam0_xyz = poses_cam0[:, :3, 3]
    #
    # poses3gpsdxyz = open3d.geometry.PointCloud()
    # poses3gpsdxyz.points = open3d.utility.Vector3dVector(poses_gps_xyz)
    # poses3gpsdxyz.paint_uniform_color([0,0,0])
    #
    #
    # poses3dcam0xyz = open3d.geometry.PointCloud()
    # poses3dcam0xyz.points = open3d.utility.Vector3dVector(poses_cam0_xyz)
    # poses3dcam0xyz.paint_uniform_color([1, 0, 0])
    #
    #
    #

    # pcd = pcd[:,:3]
    # pointcloudopen3d = open3d.geometry.PointCloud()
    # pointcloudopen3d.points = open3d.utility.Vector3dVector(pcd)
    #
    # open3d.visualization.draw_geometries([pointcloudopen3d])
    #



    return pcd


# 4x4
def get_system_pose_of_accumulated_pcd(self):
    gps_pose_file = os.path.join('../../../accumulated_kitti/2013_05_28_drive_0000_sync_005900_006000/lidar_pose.dat')
    gps_poses = np.loadtxt(gps_pose_file)
    poses_gps = np.reshape(gps_poses[:, 1:], (-1, 4, 4))
    frames = gps_poses[:, 0]

    return poses_gps, frames



def convert_accumulated_pcd_to_cam(self ,accumulated_points_world, frame):
    # rec_cam0 ----> world
    pose = self.get_rec_cam0_to_world(frame)

    # world -----> rec_cam0
    pose_inv = np.linalg.inv(pose)

    # points in rec_cam0
    accumulated_points_rec_cam = (pose_inv @ accumulated_points_world.T).T

    accumulated_points_rec_cam = accumulated_points_rec_cam[:, :3]


    u, v, depth = self.project_velodyne_in_rec_cam_to_image_space(self.camera, accumulated_points_rec_cam)


    # prepare depth map for visualization
    depthMap = np.zeros((self.camera.height, self.camera.width))
    # get mask
    mask = self.get_mask(u, v, depth, self.camera)
    depthMap[v[mask], u[mask]] = depth[mask]


    ##############################
    frustrum = accumulated_points_world[mask]
    frustrum = frustrum[:, :3]
    pointcloudopen3d = open3d.geometry.PointCloud()
    pointcloudopen3d.points = open3d.utility.Vector3dVector(frustrum)
    #
    open3d.visualization.draw_geometries([pointcloudopen3d])


    layout = (2, 1)
    fig, axs = plt.subplots(*layout, figsize=(18, 12))

    image_path = self.get_image_path(frame)

    # color map for visualizing depth map
    cm = plt.get_cmap('jet')

    colorImage = np.array(Image.open(image_path)) / 255.
    depthImage = cm(depthMap / depthMap.max())[..., :3]
    colorImage[depthMap > 0] = depthImage[depthMap > 0]

    axs[0].imshow(depthMap, cmap='jet')
    axs[0].title.set_text('Projected Depth')
    axs[0].axis('off')
    axs[1].imshow(colorImage)
    axs[1].title.set_text('Projected Depth Overlaid on Image')
    axs[1].axis('off')
    plt.suptitle('Sequence %04d, Camera %02d, Frame %010d' % (seq, cam_id, frame))
    plt.show()




 def load_sick_data(self, frame=5930):

        load_velodyne_time = np.loadtxt(self.raw3DPcdTimePath + '/timestamps.txt', delimiter='\n', converters={0: self.parsetime})
        print(load_velodyne_time.shape)
        velodyne_start_time = load_velodyne_time[frame]
        velodyne_end_time = load_velodyne_time[frame+1]


        load_sick_time = np.loadtxt(self.raw3DSickTimePath + '/timestamps.txt', delimiter='\n', converters={0: self.parsetime})
        print(load_sick_time.shape)

        mask = np.logical_and(load_sick_time > velodyne_start_time, load_sick_time < velodyne_end_time)
        indices = np.where(mask)
        indices = np.array(indices[0])

        sick_data = []
        for frame in indices:
            pcdFile = os.path.join(self.raw3DSickPath, '%010d.bin' % frame)
            if not os.path.isfile(pcdFile):
                raise RuntimeError('%s does not exist!' % pcdFile)
            pcd = np.fromfile(pcdFile, dtype=np.float32)
            pcd = np.reshape(pcd, [-1, 2])
            pcd = np.concatenate([np.zeros_like(pcd[:, 0:1]), -pcd[:, 0:1], pcd[:, 1:2]], axis=1)
            data = np.zeros((pcd.shape[0], 4))
            data[:, :3] = pcd
            data[:, 3] = 1
            print(data.shape)
            sick_data.append(data)

        sick_all = sick_data[0]
        for sick in sick_data[1:]:
            sick_all = np.concatenate((sick_all, sick), axis=0)


        print(sick_all[0, :])
        print(sick_all[1, :])
        print(sick_all[2, :])

        TrSickToVelo = self.get_sick_to_velo()

        sick_all_velo = (TrSickToVelo @ sick_all.T)
        return sick_all


def get_sick_data_in_world_coord(self, sick, frame):
    TrSickToVelo = loadCalibrationRigid(self.fileSickToVelo)
    pcd = (TrSickToVelo @ sick.T).T

    TrCam0ToVelo = loadCalibrationRigid(self.fileCameraToVelo)
    TrVeloToCam = np.linalg.inv(TrCam0ToVelo)
    TrCamToPose = self.camera.camToPose
    posestxt = self.get_gps_to_world(frame)

    pcd = (posestxt @ TrCamToPose @ TrVeloToCam @ pcd.T).T

    pcd = pcd[:, :3]
    return pcd


    def parsetime(self, txt):
        import datetime

        txt = txt.decode('ascii')
        txt = txt[:-3]

        return np.datetime64(
            datetime.datetime.strptime(txt, '%Y-%m-%d %H:%M:%S.%f')
        )