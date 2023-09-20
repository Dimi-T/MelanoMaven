import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sb
from tqdm import tqdm
from model import VGG16Model
from melanomavendataset import MelanoMavenDataset


def test(model, batch_size = 16, criterion = None, test_ds = None, num_classes = 2, threshold = 0.5):

    TP, TN, FP, FN = 0, 0, 0, 0

    test_loss = 0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    model.eval()

    for data, target in tqdm(test_loader):

        data, target = data.cuda(), target.cuda()

        output = model(data)

        loss = criterion(output, target)
        test_loss += loss.item()

        probabilities = torch.softmax(output, dim=1)

        for i in range(len(probabilities)):

            label = target.data[i]
            class_total[label] += 1

            for i in range(len(probabilities)):

                label = target.data[i]
                class_total[label] += 1

                if probabilities[i, 1] >= threshold:  # Check if the predicted probability for the positive class is above the threshold
                    if label == 1:
                        TP += 1
                        class_correct[1] += 1
                    else:
                        FP += 1
                else:
                    if label == 0:
                        TN += 1
                        class_correct[0] += 1
                    else:
                        FN += 1

    test_loss = test_loss/len(test_loader.dataset)

    return test_loss, class_correct, class_total, TP, TN, FP, FN

def main():


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG16Model().to(device)
    model.load_state_dict(torch.load('../../models/MelanoMaven v2/MelanoMaven_VGG16_v2.pt'))

    dataset = MelanoMavenDataset()
    classes = ['Benign', 'Malignant']

    criterion = nn.CrossEntropyLoss()
    batch_size = 16

    threshold = 0.45

    test_ds = dataset.test_ds

    test_loss, class_correct, class_total, TP, TN, FP, FN = test(model, batch_size = batch_size, criterion = criterion,
                                                                test_ds = test_ds, num_classes = len(classes), threshold = threshold)

    print(f"\n\nTest Loss: {test_loss:.4f}\n")
    for i in range(len(classes)):

        if class_total[i] > 0:
            print(f"Test Accuracy of {classes[i]}: {(100 * class_correct[i] / class_total[i]):.4f}%")
            print(f"{classes[i]} Prediction Accuracy: {np.sum(class_correct[i])}/{np.sum(class_total[i])}\n")
            

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    cm = np.array([[TP, FP], [FN, TN]])
    
    plt.subplots(figsize=(10, 10))
    sb.heatmap(cm, annot=True, fmt=".4f",
               xticklabels=[f"pred_{i}" for i in classes],
                yticklabels=[f"true_{i}" for i in classes])
    
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("True Label")

    try:
        plt.savefig("../../output/melanomaven_v2_confusion_matrix.png")
    except:
        pass

    with open("../../output/melanomaven_v2_score.txt", "w") as f:

        f.write("MelanoMaven VGG16 Model\n")
        f.write(f"Test Accuracy of {classes[0]}: {100 * class_correct[0] / class_total[0]:.4f}%\n")
        f.write(f"Test Accuracy of {classes[1]}: {100 * class_correct[1] / class_total[1]:.4f}%\n")
        f.write(f"Total Prediction Accuracy: {100 * np.sum(class_correct) / np.sum(class_total):.4f}%\n")
        f.write(f"Total Prediction Report: {np.sum(class_correct)}/{np.sum(class_total)}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Sensitivity: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1_score:.4f}\n")

if __name__ == '__main__':
    main()
