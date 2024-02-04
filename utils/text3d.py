import cv2 as cv
import numpy as np
import open3d as od
from .cloud import concat
from .datastructure import BasePlane
from .color import Color


def write_text_2d(text, font_scale=5, thickness=20):
    (width, height), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    img = np.zeros((baseline + height, width))
    cv.putText(img, text, (0, height+baseline//2), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255), thickness, cv.LINE_AA)
    return img

def write_text_3d(text, org, base_plane, text_height, text_color=None, font_scale=5, thickness=20, add_box=False, box_color=Color.GRAY, vertical=False):
    img = write_text_2d(text, font_scale, thickness)
    text3d = image2text3d(img, base_plane, text_height, text_color=text_color, add_box=add_box, box_color=box_color)
    if vertical:
        R = text3d.get_rotation_matrix_from_xyz([0, np.pi/2, 0])
        text3d.rotate(R, center=(0, 0, 0))
    text3d.translate(org)
    return text3d

def image2pointcloud(img, base_plane, text_height, offset=False):
    height, width = img.shape
    nonzeros = cv.findNonZero(img).squeeze(axis=1).astype(np.float32)
    nonzeros[:, 0] -= width/2
    nonzeros[:, 1] -= height/2
    depth = height//6
    if offset:
        depth = int(depth*1.2)
    vmin = -depth//2
    vmax = vmin + depth
    points = []
    for v in range(vmin, vmax):
        v_col = np.ones((nonzeros.shape[0], 1))*v
        points_ = np.concatenate([nonzeros, v_col], axis=-1)
        points.append(points_)
    points = np.concatenate(points, axis=0)
    ratio = height / text_height
    points /= ratio
    if base_plane == BasePlane.XY:
        points[:, 0] *= -1
    elif base_plane == BasePlane.XZ:
        points = np.concatenate([-points[:, [0]], points[:, [2]], -points[:, [1]]], axis=-1)
    elif base_plane == BasePlane.YZ:
        points = np.concatenate([points[:, [2]], -points[:, [0]], -points[:, [1]]], axis=-1)
    else:
        raise ValueError("Wrong plane")
    pcd = od.geometry.PointCloud(points=od.utility.Vector3dVector(points))
    return pcd

def image2text3d(img, base_plane, text_height=0.1, text_color=Color.GREEN, add_box=True, point_box_step=5, box_color=Color.GRAY):
    pcd_text = image2pointcloud(img, base_plane, text_height)
    if text_color:
        pcd_text.paint_uniform_color(text_color)
    if add_box:
        img_e = np.zeros_like(img)
        height, width = img_e.shape
        idx = range(0, width, point_box_step)
        idy = [[id_] for id_ in range(0, height, point_box_step)]
        img_e[idy, idx] = 255
        pcd_box = image2pointcloud(img_e, base_plane, text_height, offset=True)
        if box_color:
            pcd_box.paint_uniform_color(box_color)
        pcd_text = concat([pcd_text, pcd_box])
    return pcd_text