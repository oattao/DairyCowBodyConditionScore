import os
import json
from pprint import pprint
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, accuracy_score
from Pnet.pointnet import PointNetRegression
from utils.data import BCSRegressionDataset
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

def test(view, run_time=0):
    with open('config.json', 'r') as fp:
        config = json.load(fp)
    naive = False
    if view == 'full':
        num_features = 8
    elif view == 'naive':
        num_features = 3
        naive = True
    elif view == 'cow':
        num_features = 3
        naive = True
    else:
        num_features = 4

    npz_path = config['npz_path']
    label_score_path = config['label_score_path']
    score_dataframe = pd.read_csv(label_score_path)
    score_dict = dict(zip(score_dataframe['cow_regno'].values, score_dataframe['bcs'].values))
    label_score_path = config['label_score_path']
    score_dataframe = pd.read_csv(label_score_path)
    list_cow_regno = score_dataframe[score_dataframe['type']=='test']['cow_regno'].values
    if view != 'naive':
        list_npz = [f"{npz_path}/{view}/{cow_regno}.npz" for cow_regno in list_cow_regno]
    else:
        list_npz = [f"{npz_path}/center/{cow_regno}.npz" for cow_regno in list_cow_regno]
    dataset = BCSRegressionDataset(list_npz, score_dict, naive=naive) 

    dataset = BCSRegressionDataset(list_npz, score_dict, naive, 'test') 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], drop_last=False) 
    device = torch.device("cuda:0")
    model = PointNetRegression(k=num_features)
    model.to(device)
    previous_model_path = f"./trained_models/{view}_{config['st']}_{run_time}.pth"
    model.load_state_dict(torch.load(previous_model_path))
    colname = f"run_time_{run_time}"
    metrics, prediction = test_model(model, dataloader, device, colname)
    metrics_path = './log/pov_prediction.csv'
    if os.path.exists(metrics_path):
        prediction.pop('true_bcs')
        df = pd.read_csv(metrics_path, index_col='cow_regno')
        if colname in df.columns:
            df.pop(colname)
        prediction = pd.concat([df, prediction], axis=1, ignore_index=False)
    prediction.to_csv(metrics_path, index='cow_regno')

    pprint(metrics)
    # plt.plot(prediction['true_bcs'])
    # plt.plot(prediction[colname])
    # plt.legend(['True', 'Prediction'])
    # plt.show()

def main(run_time):
    # for view in ["naive", "head", "tail", "left", "right", "center", "full"]:
    view = 'full'
    print("Testing: ", view)
    test(view, run_time=i)

if __name__ == "__main__":
    for i in range(5):
        main(run_time=i)


