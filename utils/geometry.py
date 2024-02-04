import numpy as np
import open3d as od


def rodrigues_rotate(pcd, n0, n1):
    points = np.asarray(pcd.points)
    npoints = rodrigues_rotate_points(points, n0, n1)
    pcd_n = od.geometry.PointCloud()
    pcd_n.points = od.utility.Vector3dVector(npoints)
    return pcd_n

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