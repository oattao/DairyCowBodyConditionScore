import os
import torch
from sklearn.model_selection import train_test_split

def train_model(model, dataloader, optimizer, criterion, scheduler):
    model.train()
    epoch_loss = 0.0
    for i in range(len(dataloader)):
        optimizer.zero_grad()
        [verts, labels] = dataloader[i]
        with torch.set_grad_enabled(True):
            output = model.forward(verts)
        loss = criterion(output.x, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    return epoch_loss

def eval_model(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0.0
    for i in range(len(dataloader)):
        [verts, labels] = dataloader[i]
        with torch.set_grad_enabled(False):
            output = model.forward(verts)
        loss = criterion(output.x, labels)
        epoch_loss += loss.item()
    return epoch_loss

def prepare_dataloader(data_path, test_size, random_state, num_point, eps, device, PointcloudDataset):
    inputs_paths = os.listdir(data_path)
    inputs_paths = [fname for fname in inputs_paths if fname.endswith('.npz')]
    train_paths, test_paths = train_test_split(
        inputs_paths, 
        test_size=test_size, 
        random_state=random_state
    )
    print("Num train: ", len(train_paths), "Num test: ", len(test_paths))
    data_loader_training = PointcloudDataset(
        data_path, 
        train_paths, 
        eps, 
        device, 
        augmentation=False,
        num_keypoints=num_point
    )
    data_loader_testing = PointcloudDataset(
        data_path, 
        test_paths, 
        eps, 
        device, 
        augmentation=False,
        num_keypoints=num_point
    )
    dataloader = {
        'train': data_loader_training,
        'test': data_loader_testing
    }
    return dataloader