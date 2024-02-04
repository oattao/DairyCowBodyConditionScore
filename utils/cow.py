import os
import numpy as np
import cv2 as cv
import open3d as od
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from skimage.measure import LineModelND, ransac
from .datastructure import Axis, Plane, Line
from .cloud import project_points_to_standard_plane, find_pcd_near_plane, pca2d
from .registration import find_transformation
  

def show3d(pcd):
    if isinstance(pcd, np.ndarray):
        pcd = od.geometry.PointCloud(points=od.utility.Vector3dVector(pcd))
    od.visualization.draw_geometries([pcd])

def show2d(image):
    plt.imshow(image)
    plt.show()

def find_leg_peak(half_body, standard_leg, standard_leg_peak):
    # find transform 
    transformation, bbox, *_ = find_transformation(half_body, standard_leg, 0.01)
    half_body_tf = half_body.transform(transformation.transformation)
    tree = od.geometry.KDTreeFlann(half_body_tf)      
    [k, idx, _] = tree.search_knn_vector_3d(standard_leg_peak, 5)
    p = np.asarray(half_body_tf.points)[idx].mean(axis=0)
    return p, half_body_tf

def find_back_peak_more_simple(points, k=10):
    values = points[:, 2]
    indices = np.argsort(values)[::-1][:k]
    return indices  

def find_back_peak_simple(umask, umap, k=10):
    ys, xs = np.where(umask > 0)
    xs = xs.tolist()
    xs_sorted = sorted(xs)
    x_indices = []
    for x_ in xs_sorted[:k]:
        idx = xs.index(x_)
        x_indices.append(idx)
    
    back_peak_points_indices = [umap[ys[idx], xs[idx]]  for idx in x_indices]
    return back_peak_points_indices

def find_back_peak(umask, umap, halfrow):
    y, x = np.where(umask > 0)
    sorted_incides = x.argsort()
    xx = x[sorted_incides]
    yy = halfrow - y[sorted_incides]
    peaks, _ = find_peaks(yy, halfrow-5, distance=200, width=20)
    # breakpoint()
    if len(peaks) == 0:
        return None
    
    peaks_base_left = min(peaks)
    peaks_base_right = max(peaks)
    ys = halfrow - yy[peaks_base_left: peaks_base_right+1]
    xs = xx[peaks_base_left: peaks_base_right+1]
    back_peak_points_indices = [umap[y, x] for (y, x) in zip(ys, xs)]
    return back_peak_points_indices

def find_back_line(cow_body):
    box = cow_body.get_axis_aligned_bounding_box()
    xmin = box.get_min_bound()[0]
    xmax = box.get_max_bound()[0]
    xdelta = xmax - xmin
    # ymax -= ydelta*0.1
    # ymin += ydelta*0.1
    e2s = np.linspace(xmin, xmax, 50)
    back_line_points = []
    v1 = np.array([1, 0, 0])
    for e2s_ in e2s:
        center = [e2s_, 0, 0]
        belly_plane = Plane(v1, center)
        belly_contour = find_pcd_near_plane(belly_plane, cow_body, 0.02)
        belly_points = np.asarray(belly_contour.points)
        if belly_points.shape[0] < 100:
            continue
        back_peak_points_indices = find_back_peak_more_simple(belly_points)
        if back_peak_points_indices is not None:
            back_peak_points = belly_points[back_peak_points_indices]
            back_line_points.append(back_peak_points)

    if len(back_line_points) == 0:
        return None
    return np.concatenate(back_line_points)

def find_belly_line(cow_body):
    box = cow_body.get_axis_aligned_bounding_box()
    ymin = box.get_min_bound()[1]
    ymax = box.get_max_bound()[1]
    ydelta = ymax - ymin
    ymax -= ydelta*0.5
    ymin += ydelta*0.02
    e2s = np.linspace(ymin, ymax, 500)
    belly_line_points = []
    v1 = np.array([0, 1, 0])
    for e2s_ in e2s:
        center = [0, e2s_, 0]
        belly_plane = Plane(v1, center)
        belly_contour = find_pcd_near_plane(belly_plane, cow_body, 0.02)
        belly_points = np.asarray(belly_contour.points)
        if belly_points.shape[0] < 20:
            continue
        index_map, mask = project_points_to_standard_plane(belly_points, "xz")
        back_trough_points_indices = find_belly_trough(mask, index_map)
        belly_trough_points = belly_points[back_trough_points_indices]
        belly_line_points.append(belly_trough_points)

    return np.concatenate(belly_line_points)

def find_belly_trough(mask, index_map, k=3, distance_threshold=1.5):
    points = cv.findNonZero(mask)
    points = np.squeeze(points, axis=1)
    ys, xs = np.where(mask > 0)
    ys = ys.tolist()
    ys_sorted = sorted(ys)

    x_m, y_m = 0, 0
    for y_ in ys_sorted[:k]:
        idx = ys.index(y_)
        x_m += xs[idx]
        y_m += ys[idx]
    x_m, y_m = int(x_m/k), int(y_m/k)

    center, *_ = pca2d(points)
    line = Line([x_m, y_m], center)
    # breakpoint()
    distance = line.compute_distance(points)
    points = points[distance < distance_threshold]
    points = points[points[:, 1] > mask.shape[0]//2]

    belly_trough_points_indices = [index_map[p[1], p[0]] for p in points]
    return belly_trough_points_indices

def remove_cow_noise(pcd):
    ground_params, ground_point_idx = pcd.segment_plane(
        distance_threshold=0.05, 
        ransac_n=4, 
        num_iterations=10000
    )
    ground_plane_return = pcd.select_by_index(ground_point_idx)

    plane_ground = Plane(normal=ground_params[:-1], d=ground_params[-1])
    distances_to_ground = plane_ground.compute_distance(np.asarray(pcd.points))
    near_ground_point_idx = np.where(distances_to_ground < 0.4)[0]
    not_near_ground_point_idx = np.where(distances_to_ground > 0.05)[0]
    mid_idx = list(set(near_ground_point_idx).intersection(set(not_near_ground_point_idx)))
    pcd_mid = pcd.select_by_index(mid_idx)

    verticle_plane_params, verticle_plane_points_idx = pcd_mid.segment_plane(
        distance_threshold=0.005, 
        ransac_n=3, 
        num_iterations=100000
    )

    # if len(verticle_plane_points_idx) < npoints*0.5:
    #     pcd_mid = pcd_mid.select_by_index(verticle_plane_points_idx, invert=True)
    #     verticle_plane_params, verticle_plane_points_idx = pcd_mid.segment_plane(
    #         distance_threshold=0.05, 
    #         ransac_n=4, 
    #         num_iterations=10000
    #     )

    # After finding 2 plane, remove points near planes
    plane_vertical = Plane(normal=verticle_plane_params[:-1], d=verticle_plane_params[-1])
    distances_to_vertical=plane_vertical.compute_distance(np.asarray(pcd.points))
    vertical_cow_idx = np.where(distances_to_vertical>0.15)[0]
    horizontal_cow_idx = np.where(distances_to_ground>0.1)[0]
    cow_idx = list(set(vertical_cow_idx).intersection(set(horizontal_cow_idx)))
    pcd_out = pcd.select_by_index(cow_idx)

    # Remove points by staticstical
    _, cow_idx = pcd_out.remove_radius_outlier(nb_points=50, radius=0.05)
    pcd_out = pcd_out.select_by_index(cow_idx)

    # # Remove points by DBScan
    with od.utility.VerbosityContextManager(od.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd_out.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
    label_name, label_count = np.unique(labels, return_counts=True)
    label_cow = label_name[label_count.argmax()]
    index_cow = np.where(labels == label_cow)[0]
    pcd_out = pcd_out.select_by_index(index_cow)
    
    return pcd_out, ground_params, verticle_plane_params, ground_plane_return

def get_cow_regno(file_path):
    # breakpoint()
    fname = os.path.basename(file_path)
    cow_regno = fname.split('.')[0]
    cow_regno = int(cow_regno)
    return cow_regno

def get_points_from_annotation_file(file_path):
    with open(file_path, 'r') as fp:
        text = fp.read()
    parts = text.strip().split('\n')
    points = [list(map(float, part.split(', '))) for part in parts]
    return np.array(points)