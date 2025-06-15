import torch
import torchvision.models as models
from torch import nn
from torch import optim
from dataloader import train_loader, test_loader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from models import *
import argparse
# import wandb

def parse_args():
    """定义命令行参数解析器"""
    parser = argparse.ArgumentParser(description="PyTorch模型训练脚本")
    
    # 必需参数
    parser.add_argument("--model", type=str, required=True,
                       choices=["ResNet", "ViT_B_16", "ViT_H_14"])
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--t_max", type=int, default=100)
    parser.add_argument("--eta_min", type=float, default=1e-5)
    
    return parser.parse_args()

args = parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_map = {
        "ResNet": ResNet,
        "ViT_B_16": ViT_B_16,
        "ViT_H_14": ViT_H_14
}
model = model_map[args.model]().to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(DEVICE)  
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.eta_min)  # T_max为周期长度

epochs = args.epochs  
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
    # wandb.log(
    #     {
    #         "Test accuracy": test_accuracy / len(test_loader),
    #     }
    # )

    print(f"Epoch: {epoch}.. "
        f"Test loss: {test_loss / len(test_loader):.3f}.. "
        f"Test accuracy: {test_accuracy / len(test_loader):.3f}")