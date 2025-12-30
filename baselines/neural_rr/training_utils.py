from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from typing import Literal
from evaluation_metrics import AllocationMetric

# Train model
def train_model(model:nn.Module, 
                train_loader:DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer, 
                device:Literal['cpu', 'cuda'],
                epochs:int):
    model = model.to(device=device)
    model.train()
    
    running_loss = 0.0
    for epoch in range(epochs):
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True) as progress_bar:
            running_loss = 0.0
            cumulative_sample_num = 0
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs=outputs, labels=labels, valuations=inputs)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                cumulative_sample_num += inputs.size(0)
                current_loss = running_loss / cumulative_sample_num
                progress_bar.set_postfix({'loss': current_loss})

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    return running_loss / len(train_loader)

# validate model
def validate_model(model: nn.Module, 
                   val_loader: DataLoader,
                   criterion: nn.Module,
                   device:Literal['cpu', 'cuda']):
    model = model.to(device=device)
    model.eval()
    val_loss = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs=outputs, labels=labels, valuations=inputs)
            val_loss.append(loss.item())
            
    return val_loss

# test model
def test_model(model: nn.Module,
               test_loader: DataLoader,
               metric: AllocationMetric,
               device: Literal['cpu', 'cuda']):
    
    model = model.to(device=device)
    model.eval()
    test_metrics = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Test'):
            inputs, labels = inputs.to(device=device), labels.to(device)
            inputs, labels = inputs.squeeze(0), labels.squeeze(0)
            outputs = model.predict(inputs)
            test_metrics += metric.forward(inputs, outputs, labels)

    return test_metrics
