import torch
from torchvision import transforms
from model import VGG16Model
from PIL import Image

import sys


def get_predict(image_path = None, threshold = 0.5):

    if image_path is None:
        return 'No image path provided.'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG16Model().to(device)
    model.load_state_dict(torch.load('../../models/MelanoMaven v2/MelanoMaven_VGG16_v2.pt'))

    transform = transforms.Compose([
        transforms.Resize((64, 64)),                                                # Resize the image to VGG19 input size
        transforms.ToTensor(),                                                      # Convert image to tensor
        transforms.Normalize([0.485, 0.456, 0.406],                                 # Normalize image data to Imagenet standards
                            [0.229, 0.224, 0.225])
    ])

    model.eval()

    img = Image.open(image_path)
    img = transform(img)
    img = img.cuda() if torch.cuda.is_available() else img

    img = img.unsqueeze(0)
    output = model(img)
    pred =  torch.softmax(output, dim=1)
    if pred[0, 1] >= threshold:
        pred = 'Malignant'
    else:
        pred = 'Benign'
    print(f"Prediction: {pred}")
    return pred

if __name__ == '__main__':
    get_predict(sys.argv[1])