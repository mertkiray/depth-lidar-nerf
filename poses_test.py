import copy

import numpy as np
import open3d

from preprocess.KITTI360.Kitti360Dataset import Kitti360Dataset


dataset = Kitti360Dataset(seq=2, cam_id=0, kitti360Path='data/kitti/KITTI-360/')


poses_ourt = np.load('poses_our.npy')
poses_our = poses_ourt[:, : , 3]


pcd_our = open3d.geometry.PointCloud()
pcd_our.points = open3d.utility.Vector3dVector(poses_our)
pcd_our.paint_uniform_color([1, 0, 0])




pointcloud = dataset.get_velodyne_in_world_coord(6290)
pointcloud = pointcloud[:,:3]

pointcloudfrustrum = dataset.get_points_visible_in_world_coord(6290)

pointcloudopen3d = open3d.geometry.PointCloud()
pointcloudopen3dfrustrum = open3d.geometry.PointCloud()
pointcloudopen3d2 = open3d.geometry.PointCloud()
colmappcd = open3d.geometry.PointCloud()
render_poses_pcd = open3d.geometry.PointCloud()
render_poses_colmap_pcd = open3d.geometry.PointCloud()


# colmap_pointcloud = np.load('colmap_pointcloud.npy')
# colmap_pointcloud = np.array(colmap_pointcloud)


# render_poses = np.load('render_poses.npy')
# render_poses = np.array(render_poses)
# render_poses = render_poses[:, : , 3]

print(pointcloudfrustrum.shape)
pointcloudfrustrum = pointcloudfrustrum[:, :3]


pointcloudopen3dfrustrum.points = open3d.utility.Vector3dVector(pointcloudfrustrum)
pointcloudopen3d.points = open3d.utility.Vector3dVector(pointcloud)

# render_poses_pcd.points = open3d.utility.Vector3dVector(render_poses)
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


open3d.visualization.draw_geometries([pcd_our,pointcloudopen3d, mesh_frame])
#open3d.visualization.draw_geometries([pcd_our, pointcloudopen3dfrustrum, mesh_frame])
#open3d.visualization.draw_geometries([render_poses_pcd, render_poses_colmap_pcd])
#open3d.visualization.draw_geometries([render_poses_pcd, colmappcd])
