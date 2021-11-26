import copy

import numpy as np
import open3d

from preprocess.KITTI360.Kitti360Dataset import Kitti360Dataset


dataset = Kitti360Dataset(seq=0, cam_id=0, kitti360Path='data/kitti/KITTI-360/')


# pose = dataset.get_pose_of_frame(5930)
# pose = np.linalg.inv(pose)
#
# print(pose.shape)
#
# pose = pose[:3,3].reshape(1,3)
#
# print(pose.shape)

#
poses_ourt = np.load('poses_our.npy')
poses_colmapt = np.load('poses_colmap.npy')
#
#
poses_our = poses_ourt[:, : , 3]
#
poses_colmap = poses_colmapt[:, : , 3]


#our_avg = np.mean(poses_our,axis=0)
#col_avg = np.mean(poses_colmap,axis=0)

#poses_our = poses_our - our_avg
#print(poses_our.shape)
#poses_our = poses_our[0,:].reshape(1,3)
#print(poses_our.shape)
#poses_colmap = poses_colmap - col_avg


pcd_our = open3d.geometry.PointCloud()
pcd_our.points = open3d.utility.Vector3dVector(poses_our)

pcd_our.paint_uniform_color([1, 0, 0])
#pcd_our.scale(1000000, center=pcd_our.get_center())

pcd_colmap = open3d.geometry.PointCloud()
pcd_colmap.points = open3d.utility.Vector3dVector(poses_colmap)

#pcd_colmap.paint_uniform_color([0, 0, 1])



pointcloud = dataset.get_velodyne_in_world_coord(5930)
pointcloud = pointcloud[:,:3]

#pointcloud2 = dataset.get_velodyne_in_world_coord(8000)
#pointcloud2 = pointcloud2[:,:3]


pointcloudfrustrum = dataset.get_points_visible_in_world_coord(5930)

pointcloudopen3d = open3d.geometry.PointCloud()
pointcloudopen3dfrustrum = open3d.geometry.PointCloud()
pointcloudopen3d2 = open3d.geometry.PointCloud()
colmappcd = open3d.geometry.PointCloud()
render_poses_pcd = open3d.geometry.PointCloud()
render_poses_colmap_pcd = open3d.geometry.PointCloud()


colmap_pointcloud = np.load('colmap_pointcloud.npy')
colmap_pointcloud = np.array(colmap_pointcloud)


render_poses = np.load('render_poses.npy')
render_poses = np.array(render_poses)
render_poses = render_poses[:, : , 3]


render_poses_colmap = np.load('render_poses_colmap.npy')
render_poses_colmap = np.array(render_poses_colmap)
render_poses_colmap = render_poses_colmap[:, : , 3]
print(f'render_poses_colmap: {render_poses_colmap.shape}')

print(pointcloudfrustrum.shape)
pointcloudfrustrum = pointcloudfrustrum[:, :3]

whole_pcds = []

start = 5930
for x in range(21):
    index = x + start
    pointcloud = dataset.get_velodyne_in_world_coord(index)
    pointcloud = pointcloud[:, :3]

    pointcloudopen3d = open3d.geometry.PointCloud()
    pointcloudopen3d.points = open3d.utility.Vector3dVector(pointcloud)
    c = np.random.rand(1, )
    pointcloudopen3d.paint_uniform_color([0,0,c[0]])

    whole_pcds.append(pointcloudopen3d)

whole_pcds.append(pcd_our)



pointcloudopen3dfrustrum.points = open3d.utility.Vector3dVector(pointcloudfrustrum)
pointcloudopen3d.points = open3d.utility.Vector3dVector(pointcloud)
colmappcd.points = open3d.utility.Vector3dVector(colmap_pointcloud)
render_poses_pcd.points = open3d.utility.Vector3dVector(render_poses)
render_poses_colmap_pcd.points = open3d.utility.Vector3dVector(render_poses_colmap)
#pointcloudopen3d2.points = open3d.utility.Vector3dVector(pointcloud2)

pointcloudopen3d.paint_uniform_color([0, 0, 1])
pointcloudopen3dfrustrum.paint_uniform_color([0, 0, 1])
pointcloudopen3d2.paint_uniform_color([1, 0, 0])
colmappcd.paint_uniform_color([0, 1, 0])
render_poses_pcd.paint_uniform_color([0, 1, 0])
render_poses_colmap_pcd.paint_uniform_color([1,0, 0])


print('asdasdasd')

transform_our = poses_ourt[0]
transform_our = transform_our[:3,:4]

#transform_colmap = poses_colmap[0]
#transform_colmap = transform_colmap[:3,:4]

print(poses_ourt[0])
print(poses_ourt[0].shape)
ident = np.eye(4)
ident[:3,:4] = transform_our
print(ident)


mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
#mesh_frame.rtate(R, center=(0, 0, 0))
mesh_frame.transform(ident)



#pose0 = poses_ourt[0]

#transform = np.eye(4)
#transform[:3,:4] = pose0[:,:4]


#print(transform)
#transform[:3,3] = transform[:3, 3]

#pointcloudopen3d.transform(transform)

print('-----------------------------------------------')
print(render_poses[0])
print(render_poses_colmap[0])
print('------------------------------------------------')


open3d.io.write_point_cloud("pcd_scene.ply", pointcloudopen3d)
open3d.io.write_point_cloud("pcd_frustrum.ply", pointcloudopen3dfrustrum)
open3d.io.write_point_cloud("pcd_poses.ply", pcd_our)

open3d.visualization.draw_geometries(whole_pcds)
#open3d.visualization.draw_geometries([pcd_our, pointcloudopen3dfrustrum, mesh_frame])
#open3d.visualization.draw_geometries([pcd_our, pointcloudopen3dfrustrum, mesh_frame])
#open3d.visualization.draw_geometries([render_poses_pcd, render_poses_colmap_pcd])
#open3d.visualization.draw_geometries([render_poses_pcd, colmappcd])
