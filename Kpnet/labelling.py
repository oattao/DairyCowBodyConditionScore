import sys
sys.path.append('..')
import os
import random
import glob
import open3d as o3d
import numpy as np
from plyfile import PlyData
import potpourri3d as pp3d
import pandas as pd
from os import listdir
from os.path import isfile, join
from utils.cow import get_cow_regno


sampling_size = 20000

colors = { # iterate with `keys = list(colors)` and `colors[keys[0]]`
  "red"     : [1.0, 0.0, 0.0],
  "green"   : [0.0, 1.0, 0.0],
  "blue"    : [0.0, 0.0, 1.0],
  "yellow"  : [1.0, 1.0, 0.0],
  "pink"    : [1.0, 0.0, 1.0],
  "aqua"    : [0.0, 1.0, 1.0],
  "brown"   : [0.5, 0.5, 0.1],
  "orange"  : [1.0, 0.7, 0.1],
  "purple"  : [0.9, 0.4, 0.9],
  "white"   : [1.0, 1.0, 1.0]
}

def create_sphere(center): # need to add automatic scalling
    sphere = o3d.geometry.TriangleMesh.create_sphere(0.03)
    sphere.translate(center)
    sphere.compute_vertex_normals()
    return sphere

def list_folder(path):
    instance_list = [f[:-4] for f in listdir(path) if isfile(join(path, f))]
    return instance_list

def load_pcd(path, convert_mm2m=False):
    # load the data
    plydata = PlyData.read(path)
    # split the plydata data structure
    pointcloud = {}
    pointcloud['vertices'] = np.stack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).transpose()
    if convert_mm2m:
        pointcloud['vertices'] *= 0.001
    pointcloud['normals'] = np.stack([plydata['vertex']['nx'], plydata['vertex']['ny'], plydata['vertex']['nz']]).transpose()
    # pointcloud['labels'] = plydata['vertex']['label']
    # pointcloud['curvature'] = plydata['vertex']['curvature']
    # return the pointcloud
    return pointcloud

def split_pointcloud_per_camera(pointcloud):
    camera_ids = np.unique(pointcloud['labels'])
    # build a list of pointclouds for each camera
    pointclouds_list = []
    for camera_id in camera_ids:
        pointcloud_temp = {}
        pointcloud_temp['vertices'] =  pointcloud['vertices'][pointcloud['labels'] == camera_id, :]
        pointcloud_temp['normals'] =   pointcloud['normals'][pointcloud['labels'] == camera_id, :]
        # pointcloud_temp['labels'] =    pointcloud['labels'][pointcloud['labels'] == camera_id]
        # pointcloud_temp['curvature'] = pointcloud['curvature'][pointcloud['labels'] == camera_id]
        pointclouds_list.append(pointcloud_temp)
    # return the list
    return pointclouds_list

def join_pointclouds(pointclouds_list):
    print()

def find_index_in_cloud(pointcloud, annotations):
    indexes = []
    for i in range(annotations.shape[0]):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud['vertices'])
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        [k, idx, _] = pcd_tree.search_knn_vector_3d(annotations[i,:], 1)
        indexes.append(idx[0])
    return indexes

def downsample(pointcloud_in, input_sampling_size):
    input_size = pointcloud_in['vertices'].shape[0]
    # create indexes to keep
    if input_size >= input_sampling_size:
        indexes = np.sort(random.sample(range(input_size), input_sampling_size))
    else:
        number_of_points_to_add = input_sampling_size - input_size
        indexes = np.sort(list(range(input_size)) + list(random.sample(range(input_size), number_of_points_to_add)))
    #update the pointcloud
    pointcloud_out = {}
    pointcloud_out['vertices'] =  pointcloud_in['vertices'][indexes, :]
    pointcloud_out['normals'] =   pointcloud_in['normals'][indexes, :]
    # pointcloud_out['labels'] =    pointcloud_in['labels'][indexes]
    # pointcloud_out['curvature'] = pointcloud_in['curvature'][indexes]
    return pointcloud_out

def compute_distance_on_the_manifold(pointcloud, indexes, epsilon=20):
    # set variables
    annotation_number = len(indexes)
    points = pointcloud['vertices']
    solver = pp3d.PointCloudHeatSolver(points)
    # compute the distance for each annotation
    distance_on_the_manifold = np.tile(0.0, (points.shape[0], annotation_number))
    gaussian_distance = np.tile(0.0, (points.shape[0], annotation_number))
    for i in range(annotation_number):
        # compute the distance on the manifold
        dists = solver.compute_distance(indexes[i])
        distance_on_the_manifold[:, i] = dists
        # turn into Gaussian: https://en.wikipedia.org/wiki/Radial_basis_function
        dists = dists/dists.max()
        dists = np.exp(-np.square(epsilon*dists))
        gaussian_distance[:, i] = dists
    return distance_on_the_manifold, gaussian_distance

def compute_distance_virtual_point_to_manifold(pointcloud, v_points, epsilon=20):
    # set variables
    npoints = pointcloud['vertices'].shape[0]
    points = np.zeros(shape=(npoints+1, 3))
    points[:-1, :] = pointcloud['vertices']
    points[-1, :] = v_points
    solver = pp3d.PointCloudHeatSolver(points)
    # compute the distance for each annotation
    distance_on_the_manifold = np.tile(0.0, (points.shape[0], 1))
    gaussian_distance = np.tile(0.0, (points.shape[0], 1))
    # compute the distance on the manifold
    dists = solver.compute_distance(npoints)
    distance_on_the_manifold[:, 0] = dists
    # turn into Gaussian: https://en.wikipedia.org/wiki/Radial_basis_function
    dists = dists/dists.max()
    dists = np.exp(-np.square(epsilon*dists))
    gaussian_distance[:, 0] = dists
    distance_on_the_manifold = distance_on_the_manifold[:-1, :]
    gaussian_distance = gaussian_distance[:-1, :]
    return distance_on_the_manifold, gaussian_distance

def process_labels(list_pcd, annotation_folder, npz_folder, num_points=20000, epsilon=20):
    for pcd_path in list_pcd:
        instance = get_cow_regno(pcd_path)
        annotation_path = f"{annotation_folder}/{instance}.csv"
        if not os.path.exists(annotation_path):
            continue
        print("Processing: ", instance)
        # load data
        pointcloud = load_pcd(pcd_path)
        annotations = np.asarray(pd.read_csv(annotation_path, header=None))
        new_point_cloud = downsample(pointcloud, num_points)
        indexes = find_index_in_cloud(new_point_cloud, annotations)
        distance_on_the_manifold, gaussian_distance = compute_distance_on_the_manifold(new_point_cloud, indexes, epsilon)
        np.savez(f"{npz_folder}/{instance}",
                 vertices =  new_point_cloud['vertices'],
                 normals =   new_point_cloud['normals'],
                #  labels =    new_point_cloud['labels'],
                #  curvature = new_point_cloud['curvature'],
                 distance_on_the_manifold = distance_on_the_manifold,
                 gaussian_distance = gaussian_distance,
                 indexes = indexes)
        
def process_view(pcd_path, npz_folder, num_points=20000, epsilon=20):
    # load data
    cow_regno = get_cow_regno(pcd_path)
    pointcloud = load_pcd(pcd_path)
    points = pointcloud['vertices']
    xmin, ymin, zmin = points.min(axis=0)
    xmax, ymax, zmax = points.max(axis=0)
    xmean, ymean, zmean = points.mean(axis=0)
    annotations = {
        'head': np.array([xmax, ymean, zmax]),
        'tail': np.array([xmin, ymean, zmax]),
        'left': np.array([xmean, ymax, zmax]),
        'right': np.array([xmean, ymin, zmax]),
        'center': np.array([xmean, ymean, zmax])
    }
    # breakpoint()
    new_point_cloud = downsample(pointcloud, num_points)
    for view, anno in annotations.items():
        # indexes = find_index_in_cloud(new_point_cloud, anno)
        distance_on_the_manifold, gaussian_distance = compute_distance_virtual_point_to_manifold(new_point_cloud, anno, epsilon)
        # distance_on_the_manifold, gaussian_distance = compute_distance_on_the_manifold(new_point_cloud, indexes, epsilon)
        np.savez(f"{npz_folder}/{view}/{cow_regno}",
                    vertices =  new_point_cloud['vertices'],
                    normals =   new_point_cloud['normals'],
                #  labels =    new_point_cloud['labels'],
                #  curvature = new_point_cloud['curvature'],
                    distance_on_the_manifold = distance_on_the_manifold,
                    gaussian_distance = gaussian_distance,
                    # indexes = indexes
                )

# def process_labels(folder_path, num_points=20000, epsilon=20):
#     list_pcd = glob.glob(f"{folder_path}/pointclouds/*.ply")
#     for pcd_path in list_pcd:
#         instance = os.path.basename(pcd_path).split('.')[0]
#         annotation_path = f"{folder_path}/annotations/{instance}.csv"
#         if not os.path.exists(annotation_path):
#             continue
#         print("Processing: ", instance)
#         # load data
#         pointcloud = load_pcd(pcd_path)
#         annotations = np.asarray(pd.read_csv(annotation_path, header=None))
#         new_point_cloud = downsample(pointcloud, num_points)
#         indexes = find_index_in_cloud(new_point_cloud, annotations)
#         distance_on_the_manifold, gaussian_distance = compute_distance_on_the_manifold(new_point_cloud, indexes, epsilon)
#         np.savez(f"{folder_path}/processed_data/{instance}",
#                  vertices =  new_point_cloud['vertices'],
#                  normals =   new_point_cloud['normals'],
#                 #  labels =    new_point_cloud['labels'],
#                 #  curvature = new_point_cloud['curvature'],
#                  distance_on_the_manifold = distance_on_the_manifold,
#                  gaussian_distance = gaussian_distance,
#                  indexes = indexes)

def process_one_sample(pcd_path, npoints=20000, convert_mm2m=False):
    pcd = load_pcd(pcd_path, convert_mm2m)
    new_pcd = downsample(pcd, npoints)
    verts = np.expand_dims(new_pcd['vertices'], axis=0)
    return verts

def process_labels_old(folder_path):
    instances = list_folder(folder_path + "/pointclouds")
    for instance in instances:
        # load data
        print("process: " + instance)
        pcd = o3d.io.read_point_cloud(folder_path + "/pointclouds/" + instance + ".ply")
        annotation = np.asarray(pd.read_csv(folder_path + "/annotations/" + instance + ".csv", header=None))

        # set variables
        annotation_number = annotation.shape[0]
        points = np.vstack((annotation, np.asarray(pcd.points)))
        solver = pp3d.PointCloudHeatSolver(points)

        labels = np.tile(0.0, (np.asarray(pcd.points).shape[0], annotation_number))
        labels_distances = np.tile(0.0, (np.asarray(pcd.points).shape[0], annotation_number))
        for i in range(annotation_number):
            # compute the distance
            dists = solver.compute_distance(i)
            dists = dists[annotation_number:]
            labels_distances[:, i] = dists

            # turn into Gaussian: https://en.wikipedia.org/wiki/Radial_basis_function
            epsilon = 10
            dists = dists/dists.max()
            dists = np.exp(-np.square(epsilon*dists))
            labels[:, i] = dists

        # save to folder
        np.save(folder_path + "/labels/" + instance + ".npy", labels)
        np.save(folder_path + "/labels_distances/" + instance + ".npy", labels_distances)



def visualize_labels(pcd_path, anno_path, instance):
    # load data
    pcd = o3d.io.read_point_cloud(f"{pcd_path}/{instance}.ply")
    annotation = np.asarray(pd.read_csv(f"{anno_path}/{instance}.csv", header=None))

    # set variables
    annotation_number = annotation.shape[0]
    objects_to_visualize = []
    np_colors = a = np.tile(1.0, np.asarray(pcd.points).shape)
    keys = list(colors)

    for i in range(annotation_number):
        sphere = create_sphere(annotation[i])
        sphere.paint_uniform_color(np.asarray(colors[keys[i]])*0.7)
        objects_to_visualize.append(sphere)

    pcd.colors = o3d.utility.Vector3dVector(np_colors)
    objects_to_visualize.append(pcd)
    o3d.visualization.draw_geometries(objects_to_visualize)
