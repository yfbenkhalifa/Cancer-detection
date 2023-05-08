import torch
import pandas as pd
import numpy as np
import tqdm
import torch.nn as nn

def accuracy_fn(y_true:torch.tensor, y_pred:torch.tensor) -> float:
    correct = torch.eq(y_true, y_pred).sum()
    acc = (correct / len(y_true)) * 100
    return acc

def train_binary_logits(model:nn.Module, optimizer, loss_fn, X_train, y_train, X_test, y_test, epochs):
    
    for epoch in range(epochs):
        # Set model to training mode
        model.train()

        # 1. Forward pass
        y_logits = model(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))

        # 2. Compute loss
        loss = loss_fn(y_logits, y_train)

        # 2.1 Compute accuracy
        acc = accuracy_fn(y_train, y_pred)

        # 3. Optimizer zero_grad
        optimizer.zero_grad()

        # 4. Backward pass
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # 6. Test
        model.eval()
        with torch.inference_mode():
            # 6.1 Forward pass
            test_logits = model(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            # 6.2 Compute loss
            test_loss = loss_fn(test_logits, y_test)
            # 6.3 Compute accuracy
            test_acc = accuracy_fn(y_test, test_pred)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss:.4f} | Train Acc: {acc:.2f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}")


def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, device):
    """_summary_
    Args:
        model (nn.Module): model to train
        optimizer (_type_): optimizer for training
        criterion (_type_): loss function used
        train_loader (torch.utils.data.Dataset): training dataloader
        val_loader (torch.utils.data.Dataset): Validation dataloader
        num_epochs (int): number of epochs to train over
        learning_rate (double): lerning rate for the optimizer
        device (cude.device): cpu/cuda device to train on
    Returns:
        np.array: returns np.array stats for loss
        np.array: returns np.array stats for accuracy
    """
    # Accuracy and loss stats
    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }
  


    # ----- TRAINING LOOP -----#
    print("Begin training.")
    for e in tqdm(range(1, num_epochs+1)):
        
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            
            y_train_pred = model(X_train_batch)
            
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = accuracy_fn(y_train_pred, y_train_batch)
            
            train_loss.backward()
            optimizer.step()
            
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
            
            
        # VALIDATION    
        with torch.no_grad():
            
            val_epoch_loss = 0
            val_epoch_acc = 0
            
            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                
                y_val_pred = model(X_val_batch)
                            
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = accuracy_fn(y_val_pred, y_val_batch)
                
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')
    return loss_stats, accuracy_stats

# class ModelCreator():
#     @staticmethod 
#     def createNN(input_size : int, output_size : int, hidden_layer: list):
#         model = nn.Module
#         for layer in hidden_layer:
#             model.add_module(nn.Linear(in_features=))
        
def toTensor(x : np.array, type : str = "float"):
    if type == "float":
        return torch.from_numpy(x).float()
    
class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 8)
        self.act = nn.ReLU()
        self.output = nn.Linear(8, 3)
        
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x

class PCDModel_1(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.linearBlock = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=output_shape),

            
        )
        self.initWeights()

        
    def initWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.linearBlock(x)
        return x
        

    
class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super().__init__()
        
        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
        
    
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x
    