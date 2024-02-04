import open3d as od
import numpy as np
from .cloud import points2cloud


def remove_noise_by_dbscan(pcd, min_points=10, get_points=False):
    if not isinstance(pcd, od.geometry.PointCloud):
        pcd = points2cloud(pcd)
    # Remove points by DBScan
    with od.utility.VerbosityContextManager(od.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=min_points, print_progress=True))
    label_name, label_count = np.unique(labels, return_counts=True)
    label_cow = label_name[label_count.argmax()]
    index_cow = np.where(labels == label_cow)[0]
    pcd_out = pcd.select_by_index(index_cow)
    if get_points:
        return np.asarray(pcd_out.points)
    return pcd_out