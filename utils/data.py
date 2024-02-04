import random
import torch
import open3d as od
import numpy as np
import cv2 as cv
from .cow import get_cow_regno


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dict_ply_label, npoints=4096, tp='train'):
        super(SegmentationDataset, self).__init__()
        self.dict_ply_label = dict_ply_label
        self.list_file = list(self.dict_ply_label.keys())
        self.npoints = npoints
        self.tp = tp

    def __len__(self):
        return len(self.dict_ply_label)

    def __getitem__(self, idx):
        pcd_path = self.list_file[idx]  
        pcd = od.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        points = torch.as_tensor(points.T)
        label_path = self.dict_ply_label[pcd_path]
        label = np.fromfile(label_path, dtype=np.int8)
        if self.tp == 'train':
            return points, label
        return points, label, label_path
    
    def show(self, idx=None):
        if idx is None:
            idx = random.choice(range(len(self)))
        points, labels = self[idx]
        points = points.numpy()
        cow_idx = np.where(labels==1)[0]
        not_cow_idx = np.where(labels==0)[0]
        cow = od.geometry.PointCloud()
        cow.points = od.utility.Vector3dVector(points[:, cow_idx].T)
        cow.paint_uniform_color([1, 0, 0])
        not_cow = od.geometry.PointCloud()
        not_cow.points = od.utility.Vector3dVector(points[:, not_cow_idx].T)
        not_cow.paint_uniform_color([0, 1, 0])
        # od.visualization.draw_geometries([cow, not_cow])
        return cow, not_cow
    
class HipClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, dict_ply_label, dict_ply_class, npoints=1024):
        super(HipClassificationDataset, self).__init__()
        self.dict_ply_label = dict_ply_label
        self.dict_ply_class = dict_ply_class
        self.list_file = list(self.dict_ply_label.keys())
        self.npoints = npoints

    def __len__(self):
        return len(self.dict_ply_label)

    def __getitem__(self, idx):
        pcd_path = self.list_file[idx]  
        pcd = od.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)

        label_path = self.dict_ply_label[pcd_path]
        label = np.fromfile(label_path, dtype=np.int8)

        hip_idx = np.where(label==1)[0]
        hip_points = points[hip_idx]

        selected = random.sample(range(len(hip_points)), self.npoints)
        hip_points = hip_points[selected]
        hip_points = torch.as_tensor(hip_points.T)
        score = self.dict_ply_class[pcd_path]
        return hip_points, score
    
    def show(self, idx=None):
        if idx is None:
            idx = random.choice(range(len(self)))
        points, score = self[idx]
        points = points.numpy().T

        hip = od.geometry.PointCloud()
        hip.points = od.utility.Vector3dVector(points)
        hip.paint_uniform_color([0, 1, 0])
        return hip
    
class HipRegressionDataset(torch.utils.data.Dataset):
    def __init__(self, dict_ply_label, dict_ply_score, npoints=1024):
        super(HipClassificationDataset, self).__init__()
        self.dict_ply_label = dict_ply_label
        self.dict_ply_score = dict_ply_score
        self.list_file = list(self.dict_ply_label.keys())
        self.npoints = npoints

    def __len__(self):
        return len(self.dict_ply_label)

    def __getitem__(self, idx):
        pcd_path = self.list_file[idx]  
        pcd = od.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)

        label_path = self.dict_ply_label[pcd_path]
        label = np.fromfile(label_path, dtype=np.int8)

        hip_idx = np.where(label==1)[0]
        hip_points = points[hip_idx]

        selected = random.sample(range(len(hip_points)), self.npoints)
        hip_points = hip_points[selected]
        hip_points = torch.as_tensor(hip_points.T)
        score = float(self.dict_ply_score[pcd_path])
        return hip_points, score
    
    def show(self, idx=None):
        if idx is None:
            idx = random.choice(range(len(self)))
        points, score = self[idx]
        points = points.numpy().T

        hip = od.geometry.PointCloud()
        hip.points = od.utility.Vector3dVector(points)
        hip.paint_uniform_color([0, 1, 0])
        return hip
    
class BCSRegressionDataset(torch.utils.data.Dataset):
    def __init__(self, list_file, score_dict, naive=False, tp='train', distance_name='distance_on_the_manifold'):
        super(BCSRegressionDataset, self).__init__()
        self.score_dict = score_dict
        self.list_file = list_file
        self.naive=naive
        self.tp = tp
        self.distance_name = distance_name

    def __len__(self):
        return len(self.list_file)

    def __getitem__(self, idx):
        npzpath = self.list_file[idx]
        npzfile = np.load(npzpath)
        x = npzfile['vertices']
        if not self.naive:
            distances = npzfile[self.distance_name]
            x = np.hstack([x, distances])
        x = x.T
        x = torch.as_tensor(x, dtype=torch.float32)

        cow_regno = get_cow_regno(npzpath)
        y = self.score_dict[cow_regno]
        if self.tp == 'train':
            return x, y
        return x, y, cow_regno
    
class BCSNaive(torch.utils.data.Dataset):
    def __init__(self, list_file, score_dict, naive=False, tp='train'):
        super(BCSNaive, self).__init__()
        self.score_dict = score_dict
        self.list_file = list_file
        self.tp = tp

    def __len__(self):
        return len(self.list_file)

    def __getitem__(self, idx):
        pcdpath = self.list_file[idx]
        pcd = od.io.read_point_cloud(pcdpath)
        x = np.asarray(pcd.points)
        x = x.T
        x = torch.as_tensor(x, dtype=torch.float32)
        cow_regno = get_cow_regno(pcdpath)
        y = self.score_dict.get(cow_regno)
        if self.tp == 'train':
            return x, y
        return x, y, cow_regno
    
class BCSDepthMapDataset(torch.utils.data.Dataset):
    def __init__(self, list_file, score_dict, image_shape=(256, 256), tp='train'):
        super(BCSDepthMapDataset, self).__init__()
        self.score_dict = score_dict
        self.list_file = list_file
        self.image_shape = image_shape
        self.tp = tp

    def __len__(self):
        return len(self.list_file)

    def __getitem__(self, idx):
        npzpath = self.list_file[idx]
        npzfile = np.load(npzpath)
        x = npzfile['depth_map']
        x = cv.resize(x, self.image_shape)
        x = np.expand_dims(x, axis=0)
        x = np.concatenate([x, x, x], axis=0)
        x = torch.as_tensor(x, dtype=torch.float32)

        cow_regno = get_cow_regno(npzpath)
        y = self.score_dict[cow_regno]
        if self.tp == 'train':
            return x, y
        return x, y, cow_regno