from torch import utils
from torchvision import datasets
import torchvision.transforms as T


class Dataset(utils.data.Dataset):
    def __init__(self, source_path, target_path, load_size, crop_size, limit=None):
        tfm = T.Compose([
            T.Resize((load_size, 2 * load_size)),
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.source = datasets.ImageFolder(root=source_path, transform=tfm)
        print("[INFO] Found {} images in source domain folder".format(len(self.source)))
        self.target = datasets.ImageFolder(root=target_path, transform=tfm)
        print("[INFO] Found {} images in target domain folder".format(len(self.target)))
        self.size = min(len(self.source), len(self.target)) if limit == None else limit

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 0 means only images
        return (self.source[idx][0], self.target[idx][0])