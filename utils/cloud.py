import math
import copy
import open3d as od
import numpy as np
import cv2 as cv
from .datastructure import Direction, Axis


def convert_mm2m(pcd):
    points = np.asarray(pcd.points)*0.001
    pcd_out = od.geometry.PointCloud()
    pcd_out.points = od.utility.Vector3dVector(points)
    pcd_out.colors = pcd.colors
    pcd_out.normals = pcd.normals
    return pcd_out

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    od.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
    
    

def remove_noise(pcd, nb_points=16, radius=0.05):
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return pcd.select_by_index(ind)

def pca2d(points):
    data_pts = points.astype(np.float64)
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
    # Store the center of the object
    center = (int(mean[0,0]), int(mean[0,1]))
    return center, eigenvalues, eigenvectors

def pca(points):
    if not isinstance(points, np.ndarray):
        points = np.asarray(points.points)
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(points, mean)
    center = mean[0]
    e1, e2, e3 = eigenvalues.squeeze(axis=-1)
    v1, v2, v3 = eigenvectors
    return center, eigenvalues, eigenvectors

def find_pcd_near_plane(plane, pcd, threshold=0.01):
    distance = plane.compute_distance(np.asarray(pcd.points))
    idx = np.where(distance < threshold)[0]
    pcd_near_plane = pcd.select_by_index(idx)
    return pcd_near_plane

def reduce_box(box, axis, direction, reduce_ratio=0.1):
    axis_names = [Axis.X, Axis.Y, Axis.Z]
    min_bound = dict(zip(axis_names, box.get_min_bound()))
    max_bound = dict(zip(axis_names, box.get_max_bound()))
    size = dict(zip(axis_names, box.get_extent()))
    size_reduce = size[axis] * reduce_ratio
    if direction == Direction.N:
        min_bound[axis] += size_reduce
    elif direction == Direction.P:
        max_bound[axis] -= size_reduce
    min_bound = np.array([min_bound[ax] for ax in axis_names])
    max_bound = np.array([max_bound[ax] for ax in axis_names])
    box_scaled = od.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return box_scaled

def increase_box(box, axis, direction, increase_ratio=0.1):
    axis_names = [Axis.X, Axis.Y, Axis.Z]
    min_bound = dict(zip(axis_names, box.get_min_bound()))
    max_bound = dict(zip(axis_names, box.get_max_bound()))
    size = dict(zip(axis_names, box.get_extent()))
    size_reduce = size[axis] * increase_ratio
    if direction == Direction.N:
        min_bound[axis] -= size_reduce
    elif direction == Direction.P:
        max_bound[axis] += size_reduce
    min_bound = np.array([min_bound[ax] for ax in axis_names])
    max_bound = np.array([max_bound[ax] for ax in axis_names])
    box_scaled = od.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return box_scaled

def project_points_to_standard_plane(inpoints, plane_name="xy", project_ratio=512):
    if plane_name in ["xy", "yx"]:
        a, b = 0, 1
        c = 2
    elif plane_name in ["xz", "zx"]:
        a, b = 0, 2
        c = 1
    elif plane_name in ["yz", "zy"]:
        a, b = 1, 2
        c = 0
    else:
        raise ValueError("Wrong plane name!")

    points = inpoints.copy()
    if plane_name in ["yx", "zx", "zy"]:
        points[:, c] = -points[:, c]

    min_aabb = points.min(axis=0)
    max_aabb = points.max(axis=0)
    width = int((max_aabb[a] - min_aabb[a])*project_ratio) + 1
    height = int((max_aabb[b] - min_aabb[b])*project_ratio) + 1
    index_map = np.zeros((height, width), dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    for i, point in enumerate(points):
        shift = (point - min_aabb)*project_ratio
        col = int(shift[a])
        row = int(shift[b])
        # keep the closer point
        index_map[row, col] = i
        mask[row, col] = 255

    return index_map, mask

def get_depth_map(inpoints, plane_name="xy", project_ratio=512):
    if plane_name in ["xy", "yx"]:
        a, b = 0, 1
        c = 2
    elif plane_name in ["xz", "zx"]:
        a, b = 0, 2
        c = 1
    elif plane_name in ["yz", "zy"]:
        a, b = 1, 2
        c = 0
    else:
        raise ValueError("Wrong plane name!")

    points = inpoints.copy()
    if plane_name in ["yx", "zx", "zy"]:
        points[:, c] = -points[:, c]

    min_aabb = points.min(axis=0)
    max_aabb = points.max(axis=0)
    width = int((max_aabb[a] - min_aabb[a])*project_ratio) + 1
    height = int((max_aabb[b] - min_aabb[b])*project_ratio) + 1
    index_map = np.zeros((height, width), dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    depth_map = np.zeros((height, width), dtype=np.float32)
    for i, point in enumerate(points):
        shift = (point - min_aabb)*project_ratio
        col = int(shift[a])
        row = int(shift[b])
        # keep the closer point
        index_map[row, col] = i
        mask[row, col] = 255
        depth_map[row, col] = point[2]

    return depth_map, index_map, mask

def make_sphere(radius=0.01, center=None, color=None, get_cloud=False):
    mesh = od.geometry.TriangleMesh.create_sphere(radius=radius, resolution=1000)
    if center is not None:
        mesh.translate(center)
    if color is not None:
        mesh.paint_uniform_color(color)
    if get_cloud:
        mesh = mesh.sample_points_uniformly(number_of_points=1000)
    return mesh

def make_arrow(center, normal, radius, color=None):
    cylinder_radius = radius * 0.001
    cone_radius = cylinder_radius * 20
    cylinder_height = 0.07
    cone_height = cylinder_height / 4
    resolution = 100
    cylinder_split = 4
    cone_split= 1
    mesh = od.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius, 
        cone_radius=cone_radius, 
        cylinder_height=cylinder_height,
        cone_height=cone_height,
        resolution=resolution,
        cylinder_split=cylinder_split,
        cone_split=cone_split)
    pcd = mesh.sample_points_poisson_disk(number_of_points=2000)
    pcd = rodrigues_rotate(pcd, [0, 0, 1], normal)
    pcd.translate(center)
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd

def np2cloud(points: np.array, color=None):
    pcd = od.geometry.PointCloud()
    pcd.points = od.utility.Vector3dVector(points)
    if color:
        pcd.paint_uniform_color(color)
    return pcd

def rodrigues_rotate(pcd, n0, n1):
    points = np.asarray(pcd.points)
    npoints = rodrigues_rotate_points(points, n0, n1)
    return np2cloud(np.array(npoints))

def rodrigues_rotate_points(points, n0, n1):
    n0 = n0 / np.linalg.norm(n0)
    n1 = n1 /np.linalg.norm(n1)
    k = np.cross(n0, n1)
    k = k / np.linalg.norm(k)
    theta = np.arccos(np.dot(n0, n1))
    npoints = []
    for point in points:
        npoint = point * np.cos(theta) + np.cross(k, point) * np.sin(theta) + k*np.dot(k, point)*(1-np.cos(theta))
        npoints.append(npoint)
    return np.array(npoints)

def make_unit_vector(v):
    return v / np.linalg.norm(v)

def compute_angle(v1, v2):
    v1_u = make_unit_vector(v1)
    v2_u = make_unit_vector(v2)
    angle = math.atan2(np.linalg.norm(np.cross(v1_u, v2_u)), np.dot(v1_u, v2_u))
    angle *= (180 / np.pi)
    return angle

def find_points_near_plane(plane, points, dthresh=0.025):
    distance = plane.compute_distance(points)
    inlier_point_index = distance < dthresh
    points_near = points[inlier_point_index]
    return points_near

def compute_distace_point2line(line_point_1, line_point_2, point):
    num = np.cross(point-line_point_1, point-line_point_2)
    num = np.linalg.norm(num)
    den = np.linalg.norm(line_point_2-line_point_1)
    return num/den

def concat(pcds):
    points = []
    colors = []
    for pcd in pcds:
        if pcd is None:
            continue
        points.append(np.asarray(pcd.points))
        colors.append(np.asarray(pcd.colors))
    points = np.concatenate(points, axis=0)
    colors = np.concatenate(colors, axis=0)

    pcd_out = od.geometry.PointCloud()
    pcd_out.points = od.utility.Vector3dVector(points)
    pcd_out.colors = od.utility.Vector3dVector(colors)
    return pcd_out

def intersect(points1, points2):
    pass

def crop_invert(pcd, pcdx):
    bbox = pcdx.get_axis_aligned_bounding_box()
    return crop_invert_by_bbox(pcd, bbox)

def crop_invert_by_bbox(pcd, bbox):
    inliers_indices = bbox.get_point_indices_within_bounding_box(pcd.points)    
    pcd_crop = pcd.select_by_index(inliers_indices, invert=True) #select outside points
    return pcd_crop

def points2cloud(points, color=None):
    pcd = od.geometry.PointCloud()
    pcd.points = od.utility.Vector3dVector(points)
    if color:
        pcd.paint_uniform_color(color)
    return pcd

def show_points(points):
    pcd = points2cloud(points)
    od.visualization.draw_geometries([pcd])

def select_circular_index(index_map, get_big=True):
    index = cv.findNonZero(index_map)
    index = np.squeeze(index, axis=1)
    h, w = index_map.shape
    index_midle_transformed = index - np.array([w, h])//2
    rs, phis = [], []
    for (x, y) in index_midle_transformed:
        rs.append(np.sqrt(x**2 + y**2))
        phis.append(np.arctan2(y, x) / np.pi*180)
    min_phi = min(phis)
    max_phi = max(phis)
    width = int(max_phi - min_phi) + 1
    r_array = np.empty((width), dtype=np.float32)
    r_array.fill(-np.inf) if get_big else r_array.fill(np.inf)
    r_index = np.zeros((width), dtype=np.int32)
    for i, (r, phi) in enumerate(zip(rs, phis)):
        shift = phi - min_phi
        col = int(shift)
        if get_big:
            if r > r_array[col]:
                r_index[col] = i
                r_array[col] = r
        else:
            if r < r_array[col]:
                r_index[col] = i
                r_array[col] = r
    r_selected_index = index[r_index]
    selected_index = []
    for (x, y) in r_selected_index:
        selected_index.append(index_map[y, x])
    return selected_index

def find_edge_pcd_by_projection(pcd, center=None):
    points = np.asarray(pcd.points)
    plane_names = ["xz", "zx", "yz", "zy"]
    selected_indexes = []
    # for name in plane_names:
    #     index_map = project_points_to_standard_plane(points, name, center=center)
    #     selected_index = select_index(index_map)
    #     selected_indexes.extend(selected_index)

    index_map = project_points_to_standard_plane(points, "yx")
    selected_index = select_circular_index(index_map)
    selected_indexes.extend(selected_index)
    selected_indexes = list(set(selected_indexes))
    pcd_edge = pcd.select_by_index(selected_indexes)
    return pcd_edge

def find_edge_points_by_projection(points):
    selected_indexes = []

    index_map, _ = project_points_to_standard_plane(points, "yx")
    selected_index = select_circular_index(index_map)
    selected_indexes.extend(selected_index)
    selected_indexes = list(set(selected_indexes))
    points = points[selected_indexes]
    return points

class Scaler:
    def __init__(self, min_bounds, max_bounds):
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

    def transform(self, pcd):
        points = (np.asarray(pcd.points) - self.min_bounds) / (self.max_bounds - self.min_bounds)
        pcdn = copy.deepcopy(pcd)
        pcdn.points = od.utility.Vector3dVector(points)
        return pcdn
    
    def transform_points(self, points):
        points = (points - self.min_bounds) / (self.max_bounds - self.min_bounds)
        return points
    
    def invert_transform(self, pcd):
        points = np.asarray(pcd.points) * (self.max_bounds - self.min_bounds) + self.min_bounds
        pcdn = copy.deepcopy(pcd)
        pcdn.points = od.utility.Vector3dVector(points)
        return pcdn
    
    def invert_transform_points(self, points):
        points = points * (self.max_bounds - self.min_bounds) + self.min_bounds
        return points