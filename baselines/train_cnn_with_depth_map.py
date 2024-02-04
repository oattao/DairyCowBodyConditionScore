import sys
sys.path.append('..')
import os
import json
import pandas as pd
import torch
from torchvision import models
from utils.data import BCSDepthMapDataset
import wandb


def train_model(model, dataloader, criterion, device, optimizer, scheduler=None):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            outputs = torch.squeeze(outputs, dim=-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        if scheduler:
            scheduler.step()
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def eval_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            outputs = torch.squeeze(outputs, dim=-1)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def train(view):
    with open('../config.json', 'r') as fp:
        training_config = json.load(fp)
    training_config['view'] = view
    wandb.init(
        project="Dairy Body Condition Score", 
        config=training_config
    )
    config = wandb.config
    ext = 'npz'
    
    # Prepare data
    SPLITTER = ["train", "test"]
    
    npz_path = training_config['npz_path']
    label_score_path = training_config['label_score_path']
    score_dataframe = pd.read_csv(label_score_path)
    # score_dataframe = score_dataframe[score_dataframe['type']=='train']
    list_cow_regno_train = score_dataframe[score_dataframe['type']=='train']['cow_regno'].values
    list_cow_regno_test = score_dataframe[score_dataframe['type']=='test']['cow_regno'].values
    score_dict = dict(zip(score_dataframe['cow_regno'].values, score_dataframe['bcs'].values))
    list_train = [f"{npz_path}/{view}/{cow_regno}.{ext}" for cow_regno in list_cow_regno_train]
    list_test = [f"{npz_path}/{view}/{cow_regno}.{ext}" for cow_regno in list_cow_regno_test]
    list_npz = {'train': list_train, 'test': list_test}
    dataset = {
        tp: BCSDepthMapDataset(list_npz[tp], score_dict, tp='train') 
        for tp in SPLITTER
    }
    dataloader = {
        tp: torch.utils.data.DataLoader(dataset[tp], batch_size=config.batch_size, drop_last=False) 
        for tp in SPLITTER
    }

    device = torch.device("cuda:1")
    if config['depth_map_model'] == 'vgg':
        model = models.vgg19(num_classes=1)      
    else:
        model = models.efficientnet_b0(num_classes=1)
    model.to(device)
    previous_model_path = f"./trained_models/{view}-{config['depth_map_model']}_{config.st-1}.pth"
    if os.path.exists(previous_model_path):
        model.load_state_dict(torch.load(previous_model_path))
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = None
    best_loss = float('inf')
    for epoch in range(config.num_epochs):
        train_loss = train_model(model, dataloader['train'], criterion, device, optimizer, lr_scheduler)
        val_loss = eval_model(model, dataloader['test'], criterion, device)
        wandb.log({'train_loss': train_loss, 'val_loss': val_loss})

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"./trained_models/{view}-{config['depth_map_model']}_{config.st}.pth")

def main():
    view = 'depth'
    print("Training: ", view)
    train(view)

if __name__ == "__main__":
    main()