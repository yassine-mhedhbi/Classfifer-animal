from torchvision import transforms
from PIL import Image
import torch
from io import BytesIO
IMG_SIZE = 64
LABEL = {0: 'dog',
        1: 'horse',
        2: 'elephant',
        3: 'butterfly',
        4: 'chicken',
        5: 'cat',
        6: 'cow',
        7: 'sheep',
        8: 'spider',
        9: 'squirrel'}



val_augmentation = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])


def predict(package, contents):
    img = Image.open(BytesIO(contents)).convert('RGB')
    img = val_augmentation(img)[None, :]
    animal = package['model'](img)
    animal = LABEL[torch.argmax(animal).item()]
    return animal
