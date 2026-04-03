import os

from PIL import Image
from torch.utils.data import Dataset


class AnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.file_names = self.file_names = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = self.file_names[idx]
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        try:
            label_part = img_name.split(' ')[0]
            label = int(label_part) - 1
        except Exception as e:
            print(f"Error procesando archivo {img_name}: {e}")
            label = 0

        if self.transform:
            image = self.transform(image)

        return image, label