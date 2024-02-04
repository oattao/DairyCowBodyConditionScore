import random
import torch
import numpy as np
import open3d as od


color_codes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([0.2, 0.2, 0.2])]

def predict_segmentation(model, pcd, npoints, nclasses, device):
    points = np.asarray(pcd.points)
    # selected = random.sample(range(len(points)), npoints)
    # points = points[selected]
    points_tensor = points.T
    points_tensor = np.expand_dims(points_tensor, axis=0)
    points_tensor = torch.as_tensor(points_tensor, dtype=torch.float32)
    points_tensor = points_tensor.to(device)
    
    with torch.no_grad():
        prediction = model(points_tensor)
    pred = prediction[0][0].cpu().numpy()
    pred = np.argmax(pred, axis=1).astype(np.int8)
    colors = np.zeros_like(points)
    for i in range(nclasses):
        idx = np.where(pred==i)[0]
        colors[idx] = color_codes[i]
        
    cow_pcd = od.geometry.PointCloud()
    cow_pcd.points = od.utility.Vector3dVector(points)
    cow_pcd.colors = od.utility.Vector3dVector(colors)
    return pred, cow_pcd