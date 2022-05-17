from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import zeros, float32
from torchvision import transforms


IMG_SIZE = 64
NUM_CLASSES = 10
ID_COLNAME = 'file'
ANSWER_COLNAME = 'animal'
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


class AnimalDataset(Dataset):
    def __init__(self,
                 df,
                 n_classes = NUM_CLASSES,
                 id_colname = ID_COLNAME,
                 answer_colname = ANSWER_COLNAME,
                 label_dict = LABEL,
                 transforms = None
                ):
        self.df = df
        self.n_classes = n_classes
        self.id_colname = id_colname
        self.answer_colname = answer_colname
        self.label_dict = label_dict
        self.transforms = transforms
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        cur_idx_row = self.df.iloc[idx]
        img_file = cur_idx_row[self.id_colname]
        cls = cur_idx_row[self.answer_colname]
        img = Image.open(img_file).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        if self.answer_colname is not None:              
            label = zeros((self.n_classes,), dtype=float32)
            label[cls] = 1.0

            return img, label
        
        else:
            return img, img_file

train_augmentation = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

val_augmentation = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])