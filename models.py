from torch import nn
import torchvision.models as models
import torchvision
import torch
from transformers import CLIPVisionModel, AutoProcessor

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        # 加载预训练模型
        self.backbone = models.resnet50(pretrained=True)
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
    
class ViT_B_16(nn.Module):
    def __init__(self, all_weight=False, num_classes=73):
        super().__init__()
        pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        self.vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)

        for parameter in self.vit.parameters(): # 冻结参数
            # parameter.requires_grad = all_weight
            parameter.requires_grad = True
            # parameter.requires_grad = False

        self.vit.heads = nn.Linear(in_features=768, out_features=num_classes)
    
    def forward(self, x):
        return self.vit(x)
    
class ViT_H_14(torch.nn.Module):
    def __init__(self, all_weight=False, num_classes=73):
        super().__init__()
        # 统一使用float32精度
        self.vit = CLIPVisionModel.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            torch_dtype=torch.float32  
        )
        
        for param in self.vit.parameters():
            param.requires_grad = True
            
        self.classifier = torch.nn.Linear(
            self.vit.config.hidden_size, 
            num_classes
        )
        
        self.processor = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )

    def forward(self, x):
        # 动态判断是否需要缩放
        do_rescale = not (x.dtype == torch.float32 and x.max() <= 1.0)
        
        # 预处理
        inputs = self.processor(
            images=x,
            return_tensors="pt",
            do_rescale=do_rescale  
        )
        
        inputs = inputs.to(self.vit.device)
        
        # 特征提取
        outputs = self.vit(**inputs)
        pooled_output = outputs.pooler_output 
        
        return self.classifier(pooled_output.float())  