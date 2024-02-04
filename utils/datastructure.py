import open3d as od
import numpy as np
from enum import Enum
from .geometry import rodrigues_rotate

class Axis:
    X = 0
    Y = 1
    Z = 2

class Direction:
    P = 1   # positive
    N = -1  # negative

class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self._compute()

    def _compute(self):
        x1, y1 = self.p1
        x2, y2 = self.p2
        if x1 != x2:
            self.a = (y1 - y2) / (x1 - x2)
            self.b = y1 - self.a * x1
            self.normal = np.array([self.a, -1])
        else:
            self.c = x1
            self.a = None
            self.b = None
            self.normal = np.array([0, 1])

    def compute_distance(self, points):
        if self.a is None:
            return np.abs(points[:, 0] - self.c)
        return np.abs(np.dot(points, self.normal) + self.b) / np.sqrt(np.dot(self.normal, self.normal))
    
class Line3D:
    def __init__(self, center, normal):
        self.center = center
        self.normal = normal

    def compute_distance(self):
        pass

class Plane:
    def __init__(self, normal, center=None, d=None):
        self.center = center
        self.normal = normal
        if d is None:
            self.d = -np.dot(self.center, self.normal)
        else:
            self.d = d

    def manifest(self, size, resolution):
        mesh = od.geometry.TriangleMesh.create_cylinder(radius=size, height=0.001, resolution=100, split=4)
        pcd = mesh.sample_points_poisson_disk(number_of_points=resolution)
        pcd = rodrigues_rotate(pcd, [0, 0, 1], self.normal)
        pcd.translate(self.center)
        return pcd
    
    def compute_distance(self, points):
        return np.abs(np.dot(points, self.normal) + self.d) / np.sqrt(np.dot(self.normal, self.normal))

