import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size: int = 512):
    return A.compose([
        A.Resize(img_size, img_size),
        A.HorizontalFLip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])

def get_val_transforms(img_size: int=512):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456,0.406), std=(0.229, 0.224,0.225)),
        ToTensorV2(),
    ])