import torch
import torch.nn as nn
from tqdm import tqdm
from numpy import inf
from model import VGG16Model
from melanomavendataset import MelanoMavenDataset

    

def train(model, batch_size = 16, criterion = None, optimizer = None, train_ds = None):
    
    train_loss = 0
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model.train()

    for data, target in tqdm(train_loader):

        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        l2_lambda = 0.0005                                                                      # L2 regularization for better generalization
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + l2_lambda * l2_norm
        loss.backward()

        optimizer.step()
        train_loss += loss.item()

    return train_loss/len(train_loader.dataset)

def validate(model, batch_size = 16, criterion = None, valid_ds = None):
    
    valid_loss = 0

    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=True)

    model.eval()

    for data, target in tqdm(valid_loader):

        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item()

    return valid_loss / len(valid_loader.dataset)

def build(model, epochs = 20, batch_size = 16, criterion = None, optimizer = None, train_ds = None, valid_ds = None):
    
    best_val_loss = inf

    for epoch in range(epochs):

        train_loss = train(model, batch_size, criterion, optimizer, train_ds)
        valid_loss = validate(model, batch_size, criterion, valid_ds)

        print(f'Epoch: {epoch+1}/{epochs}.. Training loss: {train_loss}.. Validation Loss: {valid_loss}')

        if valid_loss < best_val_loss:
            
            print(f"Better model found: \n\t{best_val_loss:0.4f} -> {valid_loss:0.4f} \n Saving model...")
            torch.save(model.state_dict(), 'MelanoMaven_VGG16.pt')
            best_val_loss = valid_loss


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG16Model().to(device)  
    
    dataset = MelanoMavenDataset()
    epochs, batch_size = 100, 8
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for param in model.features.parameters():                             # freeze feature layers
        param.requires_grad = False


    build(model, epochs, batch_size, criterion, optimizer, dataset.train_ds, dataset.valid_ds)


if __name__ == '__main__':
    main()