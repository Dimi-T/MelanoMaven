from torchvision import datasets
from torch.utils.data import Dataset as Dataset

class MyDataset(Dataset):

    def __init__(self, dir = '', transforms = False):
        
        self.transforms = transforms
        self.dataset = datasets.ImageFolder(dir, transform=self.transforms)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        image, label = self.dataset[idx]

        return image, label