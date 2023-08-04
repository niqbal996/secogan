from torch import utils
from torchvision import datasets
import torchvision.transforms as T


class Dataset(utils.data.Dataset):
    def __init__(self, source_path):
        tfm = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.source = datasets.ImageFolder(root=source_path, transform=tfm)
        print("[INFO] Found {} images in source domain folder".format(len(self.source)))
        self.size = len(self.source)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (self.source[idx][0], self.source.imgs[idx][0]) # image as tensor and its path