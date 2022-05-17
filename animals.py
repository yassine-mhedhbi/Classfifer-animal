import numpy as np 
import pandas as pd 
import torch
from AnimalDataset import AnimalDataset, train_augmentation, val_augmentation, NUM_CLASSES
from tools import validate, f1_score, fbeta_score, cuda, train_one_epoch, train
import neptune.new as neptune
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from keys import apikey

N_EPOCHS = 10
BS = 64
TRAIN_LOGGING_EACH = 100



df = pd.read_csv('train_csv')
train_df, test_df = train_test_split(df, test_size = 0.15, shuffle = True)

model = models.densenet121(pretrained='imagenet')
model.classifier = torch.nn.Linear(model.classifier.in_features, NUM_CLASSES)
model.cuda()

train_dataset = AnimalDataset(train_df, transforms = train_augmentation)
test_dataset = AnimalDataset(test_df, transforms = val_augmentation)
train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False, num_workers=0)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)




train_losses = []
valid_losses = []
valid_f1s = []
best_model_f1 = 0.0
best_model = None
best_model_ep = 0

for epoch in range(1, N_EPOCHS + 1):
    ep_logstr = f"Starting {epoch} epoch..."
    print(ep_logstr)
    tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, TRAIN_LOGGING_EACH)
    train_losses.append(tr_loss)
    tr_loss_logstr = f'Mean train loss: {round(tr_loss,5)}'
    print(tr_loss_logstr)
    
    valid_loss, valid_f1 = validate(model, test_loader, criterion)  
    valid_losses.append(valid_loss)    
    valid_f1s.append(valid_f1)       
    val_loss_logstr = f'Mean valid loss: {round(valid_loss,5)}'
    print(val_loss_logstr)
    sheduler.step(valid_loss)
    
    if valid_f1 >= best_model_f1:    
        best_model = model        
        best_model_f1 = valid_f1        
        best_model_ep = epoch







