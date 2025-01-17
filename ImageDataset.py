import os
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_list = [file for file in os.listdir(directory) if file.endswith('.jpeg')]
        self.categories = sorted({"_".join(file.split('_')[:-1]) for file in self.image_list})
        self.category_to_idx = {category: idx for idx, category in enumerate(self.categories)}
        self.total_classes = len(self.categories)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_file = self.image_list[index]
        img_path = os.path.join(self.directory, img_file)
        
        # Load image and apply transformations
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        # Extract label from file name
        label_name = "_".join(img_file.split('_')[:-1])
        label = self.category_to_idx[label_name]

        return img, label
