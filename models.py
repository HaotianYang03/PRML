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
    
# class CLIPViT_H14_ForPollenClassification(torch.nn.Module):
#     def __init__(self, all_weight=False, num_classes=73):
#         super().__init__()
#         # 加载CLIP-ViT-H-14模型[2](@ref)
#         self.vit = CLIPVisionModel.from_pretrained(
#             "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
#             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
#         )
#         # 冻结参数逻辑
#         for param in self.vit.parameters():
#             param.requires_grad = all_weight
            
#         # 替换分类头[1](@ref)
#         self.classifier = torch.nn.Linear(
#             self.vit.config.hidden_size, 
#             num_classes
#         )
        
#         # 图像预处理适配[2](@ref)
#         self.processor = AutoProcessor.from_pretrained(
#             "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
#         )

#     def forward(self, x):
#         # 预处理（需保持与训练时相同）
#         inputs = self.processor(images=x, return_tensors="pt")
#         outputs = self.vit(**inputs.to(self.vit.device))
#         # 获取CLS token作为分类特征[1](@ref)
#         pooled_output = outputs.pooler_output 
#         return self.classifier(pooled_output)

class ViT_H_14(torch.nn.Module):
    def __init__(self, all_weight=False, num_classes=73):
        super().__init__()
        # 统一使用float32精度[2,4](@ref)
        self.vit = CLIPVisionModel.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            torch_dtype=torch.float32  # 强制使用单精度浮点数
        )
        
        # 冻结参数逻辑
        for param in self.vit.parameters():
            # param.requires_grad = all_weight
            param.requires_grad = True
            
        # 分类头（自动继承float32类型）
        self.classifier = torch.nn.Linear(
            self.vit.config.hidden_size, 
            num_classes
        )
        
        # 图像预处理器
        self.processor = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )

    def forward(self, x):
        # 动态判断是否需要缩放（避免重复缩放）[1](@ref)
        do_rescale = not (x.dtype == torch.float32 and x.max() <= 1.0)
        
        # 预处理（保持与训练一致）
        inputs = self.processor(
            images=x,
            return_tensors="pt",
            do_rescale=do_rescale  # 智能缩放控制
        )
        
        # 确保输入与模型设备一致
        inputs = inputs.to(self.vit.device)
        
        # 特征提取
        outputs = self.vit(**inputs)
        pooled_output = outputs.pooler_output 
        
        # 确保分类头输入为float32[2](@ref)
        return self.classifier(pooled_output.float())  # 显式转换类型