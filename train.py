import torch
import torchvision.models as models
from torch import nn
from torch import optim
from dataloader import train_loader, test_loader
from tqdm import tqdm
from models import *
# import wandb

# wandb.init(
#     project="PRML",
#     name="Baseline",             
# )

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CustomResNet().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(DEVICE)  
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)

epochs = 200  
train_losses, test_losses = [], []

for epoch in range(epochs):
    model.train()
    for img, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
        labels = torch.tensor(labels, dtype=torch.long)
        img, labels = img.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()  
        outputs = model(img)
        loss = loss_fn(outputs, labels)
        loss.backward()  
        optimizer.step()  

    test_loss = 0
    test_accuracy = 0

    model.eval()
    with torch.no_grad():
        for img, labels in tqdm(test_loader, desc=f"Val Epoch {epoch + 1}/{epochs}"):
            labels = torch.tensor(labels, dtype=torch.long) 
            img, labels = img.to(DEVICE), labels.to(DEVICE)
            outputs = model(img)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)

            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Epoch: {epoch}.. "
        f"Test loss: {test_loss / len(test_loader):.3f}.. "
        f"Test accuracy: {test_accuracy / len(test_loader):.3f}")