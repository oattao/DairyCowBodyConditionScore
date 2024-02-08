import sys
sys.path.append('..')
import os
import json
from pprint import pprint
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import models
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, accuracy_score
from Pnet.pointnet import PointNetRegression
from utils.data import BCSDepthMapDataset
from utils.cow import get_cow_regno


def test_model(model, dataloader, device, colname):
    model.eval()
    true, pred, cow_regnos = [], [], []
    for inputs, labels, cow_regno in dataloader:
        inputs = inputs.to(device, dtype=torch.float)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
        true.append(labels.numpy())
        pred.append(outputs.cpu().numpy())
        cow_regnos.append(cow_regno)

    true = np.concatenate(true, axis=0)
    pred = np.concatenate(pred, axis=0)
    pred = np.squeeze(pred, axis=-1)
    pred = np.ceil(pred).astype(np.int8)
    cow_regnos = np.concatenate(cow_regnos, axis=0)
    mae = mean_absolute_error(true, pred)
    rmse = mean_squared_error(true, pred)**0.5
    mape = mean_absolute_percentage_error(true, pred)
    acc = accuracy_score(true, pred)*100
    r2 = r2_score(true, pred)
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "ACC": acc}
    prediction = pd.DataFrame({'cow_regno': cow_regnos, 'true_bcs': true, colname: pred})
    prediction = prediction.set_index('cow_regno', drop=True)
    return metrics, prediction

def test(view):
    with open('../config.json', 'r') as fp:
        config = json.load(fp)

    npz_path = config['npz_path']
    label_score_path = config['label_score_path']
    score_dataframe = pd.read_csv(label_score_path)
    list_cow_regno = score_dataframe[score_dataframe['type']=='test']['cow_regno'].values
    score_dict = dict(zip(score_dataframe['cow_regno'].values, score_dataframe['bcs'].values))
    list_npz = [f"{npz_path}/{view}/{cow_regno}.npz" for cow_regno in list_cow_regno]
    dataset = BCSDepthMapDataset(list_npz, score_dict, tp='test') 

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], drop_last=False) 
    device = torch.device("cuda:0")
    if config['depth_map_model'] == 'vgg':
        model = models.vgg19(num_classes=1)      
    else:
        model = models.efficientnet_b0(num_classes=1)
    model.to(device)
    previous_model_path = f"./trained_models/{view}-{config['depth_map_model']}_{config['st']}.pth"
    model.load_state_dict(torch.load(previous_model_path))
    colname = f"byview_{view}-{config['depth_map_model']}"
    metrics, prediction = test_model(model, dataloader, device, colname)
    metrics_path = '../log/prediction.csv'
    if os.path.exists(metrics_path):
        prediction.pop('true_bcs')
        df = pd.read_csv(metrics_path, index_col='cow_regno')
        if colname in df.columns:
            df.pop(colname)
        prediction = pd.concat([df, prediction], axis=1, ignore_index=False)
    prediction.to_csv(metrics_path, index='cow_regno')
    pprint(metrics)

def main():
    test('depth')

if __name__ == "__main__":
    main()


