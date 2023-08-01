from torch import utils
from torchvision import datasets
import torchvision.transforms as T


class Dataset(utils.data.Dataset):
    def __init__(self, source_path, target_path):
        tfm = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.source = datasets.ImageFolder(root=source_path, transform=tfm)
        self.target = datasets.ImageFolder(root=target_path, transform=tfm)
        self.size = min(len(self.source), len(self.target))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 0 means only images
        return (self.source[idx][0], self.target[idx][0])