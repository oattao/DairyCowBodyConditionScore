from copy import deepcopy
import open3d as od
import numpy as np
from .cloud import concat, rodrigues_rotate
from .datastructure import Axis


def create_sphere(radius=0.01, center=None, color=None):
    sphere = od.geometry.TriangleMesh.create_sphere(radius=radius, resolution=1000)
    if center is not None:
        sphere.translate(center)
    if color is not None:
        sphere.paint_uniform_color(color)
    sphere = sphere.sample_points_poisson_disk(number_of_points=5000)
    return sphere

def create_line(p1, p2, color=None):
    line = od.geometry.LineSet(
        points=od.utility.Vector3dVector([p1, p2]),
        lines=od.utility.Vector2iVector([[0, 1]])
    )
    if color:
        line.paint_uniform_color(color)
    return line

def create_line2(p1, p2, color=None):
    line = create_cylinder(0, np.linalg.norm(p1 - p2), 0.01, Axis.X)
    line = rodrigues_rotate(line, np.array([1, 0, 0]), p1 - p2)
    line.translate((p1+p2)*0.5)
    if color:
        line.paint_uniform_color(color)
    return line

def create_plane_xy(center, x_size, y_size, color=None):
    x, y, z = center
    xmin = x - x_size
    xmax = x + x_size
    ymin = y - y_size
    ymax = y + y_size

    p1 = [xmin, ymin, z]
    p2 = [xmin, ymax, z]
    p3 = [xmax, ymax, z]
    p4 = [xmax, ymin, z]
    plane = od.geometry.LineSet(
        points=od.utility.Vector3dVector([p1, p2, p3, p4]),
        lines=od.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]])
    )
    if color:
        plane.paint_uniform_color(color)
    return plane

def create_ground_plane(center, xsize, ysize, pipe_radius, color=None):
    pipex = create_cylinder(center, xsize, pipe_radius, Axis.X)
    pipex1 = deepcopy(pipex).translate([0, -ysize/2, 0])
    pipex2 = deepcopy(pipex).translate([0, +ysize/2, 0])
    
    pipey = create_cylinder(center, ysize, pipe_radius, Axis.Y)
    pipey1 = deepcopy(pipey).translate([-xsize/2, 0, 0])
    pipey2 = deepcopy(pipey).translate([+xsize/2, 0, 0])
    
    plane = concat([pipex, pipex1, pipex2, pipey, pipey1, pipey2])
    if color:
        plane.paint_uniform_color(color)
    return plane

def create_cylinder(center, length, radius, axis=Axis.Z, color=None, get_points=True, n_points=10000):
    cylinder = od.geometry.TriangleMesh.create_cylinder(radius, length)
    if axis != Axis.Z:
        rotation_axis = [0, 0, 0]
        rotation_axis[1-axis] = np.pi/2
        R = cylinder.get_rotation_matrix_from_xyz(rotation_axis)
        cylinder.rotate(R, center=(0, 0, 0))
    cylinder.translate(center)
    if color:
        cylinder.paint_uniform_color(color)
    if get_points:
        cylinder = cylinder.sample_points_poisson_disk(number_of_points=n_points)
    return cylinder

def cut_cylinder_at_middle(cylinder, center, length, axis, cut_ratio=0.8, color=None):
    points = np.asarray(cylinder.points)
    length *= cut_ratio
    np.logical_or
    points = points[np.logical_or(points[:, axis] < center[axis]-length/2, points[:, axis] > center[axis]+length/2)]
    cylinder = od.geometry.PointCloud(points=od.utility.Vector3dVector(points))
    if color:
        cylinder.paint_uniform_color(color)
    return cylinder

def create_box(bbox, pipe_radius=0.005, color=None, cut_middle=False, cut_ratio=0.8):
    lx, ly, lz = bbox.get_max_bound() - bbox.get_min_bound()
    center = bbox.get_center()
    x, y, z = center
    # Create X pipes
    pipex = create_cylinder(center, lx, pipe_radius, Axis.X, color, True)
    if cut_middle:
        pipex = cut_cylinder_at_middle(pipex, center, lx, Axis.X, cut_ratio, color)
    pipex1 = deepcopy(pipex).translate([0, -ly/2, -lz/2])
    pipex2 = deepcopy(pipex).translate([0, +ly/2, +lz/2])
    pipex3 = deepcopy(pipex).translate([0, -ly/2, +lz/2])
    pipex4 = deepcopy(pipex).translate([0, +ly/2, -lz/2])
    # Create Y pipes
    pipey = create_cylinder(center, ly, pipe_radius, Axis.Y, color, True)
    if cut_middle:
        pipey = cut_cylinder_at_middle(pipey, center, ly, Axis.Y, cut_ratio, color)
    pipey1 = deepcopy(pipey).translate([-lx/2, 0, -lz/2])
    pipey2 = deepcopy(pipey).translate([+lx/2, 0, +lz/2])
    pipey3 = deepcopy(pipey).translate([-lx/2, 0, +lz/2])
    pipey4 = deepcopy(pipey).translate([+lx/2, 0, -lz/2])
    # Create Z pipes
    pipez = create_cylinder(center, lz, pipe_radius, Axis.Z, color, True)
    if cut_middle:
        pipez = cut_cylinder_at_middle(pipez, center, lz, Axis.Z, cut_ratio, color)
    pipez1 = deepcopy(pipez).translate([-lx/2, -ly/2, 0])
    pipez2 = deepcopy(pipez).translate([+lx/2, +ly/2, 0])
    pipez3 = deepcopy(pipez).translate([-lx/2, +ly/2, 0])
    pipez4 = deepcopy(pipez).translate([+lx/2, -ly/2, 0])
    box = concat([pipex1, pipex2, pipex3, pipex4, pipey1, pipey2, pipey3, pipey4, pipez1, pipez2, pipez3, pipez4])
    return box
    
def create_box_complex(bbox, big_pipe_radius=0.05, small_pile_radius=0.005, big_pipe_color=None, small_pipe_color=None, cut_ratio=0.8):
    big_pipes = create_box(bbox, big_pipe_radius, big_pipe_color, cut_middle=True, cut_ratio=cut_ratio)
    small_pipes = create_box(bbox, small_pile_radius, small_pipe_color, False)
    box = concat([big_pipes, small_pipes])
    return box

def create_2head_arrow(center, length, radius, cone_radius=0.01, axis=Axis.Z, color=None, n_points=10000):
    half_length = length*0.5
    cone_height = min(0.05, 0.2*half_length)
    cylinder_height = half_length - cone_height
    resolution = 100
    cylinder_split = 4
    cone_split= 1
    arror1 = od.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius, 
        cone_radius=cone_radius, 
        cylinder_height=cylinder_height,
        cone_height=cone_height,
        resolution=resolution,
        cylinder_split=cylinder_split,
        cone_split=cone_split
    )
    arrow1 = arror1.sample_points_poisson_disk(number_of_points=n_points)
    points1 = np.asarray(arrow1.points)
    points2 = deepcopy(points1)
    points2[:, 2]*=-1
    points = np.concatenate([points1, points2], axis=0)
    arrow = od.geometry.PointCloud(points=od.utility.Vector3dVector(points))
    if axis != Axis.Z:
        rotation_axis = [0, 0, 0]
        rotation_axis[1-axis] = np.pi/2
        R = arrow.get_rotation_matrix_from_xyz(rotation_axis)
        arrow.rotate(R, center=(0, 0, 0))
    arrow.translate(center) 
    if color:
        arrow.paint_uniform_color(color)
    return arrow