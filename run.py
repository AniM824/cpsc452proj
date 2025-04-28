from data import RotatedCIFAR10
from baseline import BaselineInceptionResnetV1
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])
dataset = RotatedCIFAR10(root="./data", train=True, download=True, transform=transform, rotation_angle=None)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = BaselineInceptionResnetV1(pretrained=None, classify=True, num_classes=10, device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        current_loss = running_loss / total
        current_acc = 100. * correct / total
        progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    print(f"Epoch {epoch+1:02d} | Final Loss: {epoch_loss:.4f} | Final Accuracy: {epoch_acc:.2f}%")
