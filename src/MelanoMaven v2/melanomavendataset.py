import torchvision.transforms as transforms
from dataset import MyDataset

class MelanoMavenDataset:

    def __init__(self):
        
        transform = transforms.Compose([
            transforms.Resize((64, 64)),                                              # Resize the image to VGG19 input size
            transforms.ToTensor(),                                                      # Convert image to tensor
            transforms.Normalize([0.485, 0.456, 0.406],                                 # Normalize image data to Imagenet standards
                             [0.229, 0.224, 0.225])            
        ])

        self.train_ds = MyDataset(dir = '../../datav2/train/', transforms = transform)
        self.valid_ds = MyDataset(dir = '../../datav2/valid/', transforms = transform)
        self.test_ds = MyDataset(dir = '../../datav2/test/', transforms = transform)
        