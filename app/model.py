from torchvision.models import densenet121
import torch 
NUM_CLASSES = 10

def Model():
    model = densenet121(pretrained='imagenet')
    model.classifier = torch.nn.Linear(model.classifier.in_features, NUM_CLASSES)
    return model