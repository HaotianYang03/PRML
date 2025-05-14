from torch import nn
import torchvision.models as models

class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()

        # 加载预训练ResNet50
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = nn.Identity()  

        # 自定义全连接层
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256), 
            nn.ReLU(),
            nn.Linear(256, 73),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, img):
        features = self.backbone(img)
        return self.classifier(features)