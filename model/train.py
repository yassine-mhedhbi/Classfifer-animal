import numpy as np 
import pandas as pd 
import torch
from AnimalDataset import AnimalDataset, train_augmentation, val_augmentation, NUM_CLASSES
from tools import validate, train_one_epoch
import neptune.new as neptune
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from keys import API_KEY, PROJECT

N_EPOCHS = 10
BS = 64
TRAIN_LOGGING_EACH = 100

def main(lr = 0.0005):
    run = neptune.init_model_version(
        model="AN-MOD",
        project=PROJECT,
        api_token=API_KEY,
    )  # your credentials

    df = pd.read_csv('train.csv')
    train_df, test_df = train_test_split(df, test_size = 0.15, shuffle = True)

    model = models.densenet121(pretrained='imagenet')
    model.classifier = torch.nn.Linear(model.classifier.in_features, NUM_CLASSES)
    model.cuda()

    train_dataset = AnimalDataset(train_df, transforms = train_augmentation)
    test_dataset = AnimalDataset(test_df, transforms = val_augmentation)
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False, num_workers=0)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    params = {"Model": "densenet121",
              "learning_rate": lr, 
              "optimizer": "Adam",
              "batch_size": BS,
              "n_epochs": N_EPOCHS}
    run["parameters"] = params

    best_model_f1 = 0.0
    best_model = None

    for epoch in range(1, N_EPOCHS + 1):
        ep_logstr = f"Starting {epoch} epoch..."
        print(ep_logstr)
        
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, TRAIN_LOGGING_EACH)
        run[f"train/loss"].log(round(tr_loss,5))
        tr_loss_logstr = f'Mean train loss: {round(tr_loss,5)}'
        print(tr_loss_logstr)
        
        valid_loss, valid_f1 = validate(model, test_loader, criterion)  
        run["eval/loss"].log(valid_loss)    
        run["eval/f1_score"].log(valid_f1)       
        val_loss_logstr = f'Mean valid loss: {round(valid_loss,5)}'
        print(val_loss_logstr)
        
        sheduler.step(valid_loss)
        if valid_f1 >= best_model_f1:    
            best_model = model        
            best_model_f1 = valid_f1 
            run["eval/best_f1_score"] = best_model_f1       
            run["eval/best_epoch"] = epoch 
                    
    torch.save(best_model.state_dict(), 'model_weights.pt')
    run["model/saved_model"].upload('model_weights.pt')
    run.stop()
    
    
    
if __name__ == '__main__':
    main()







