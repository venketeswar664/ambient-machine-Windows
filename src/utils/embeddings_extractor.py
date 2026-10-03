import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

class AttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor):
        super().__init__()
        self.up_factor = up_factor
        self.W_l = nn.Conv2d(in_features_l, attn_features, kernel_size=1, bias=False)
        self.W_g = nn.Conv2d(in_features_g, attn_features, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(attn_features, 1, kernel_size=1, bias=True)

    def forward(self, l, g):
        N, C, H, W = l.size()
        l_ = self.W_l(l)
        g_ = F.interpolate(self.W_g(g), scale_factor=self.up_factor, 
                         mode='bilinear', align_corners=False)
        c = self.phi(F.relu(l_ + g_))
        a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, H, W)
        return (a.expand_as(l) * l).view(N, C, -1).sum(dim=2)

class AttnResNet(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.conv1 = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1  # 256
        self.layer2 = net.layer2  # 512
        self.layer3 = net.layer3  # 1024
        self.layer4 = net.layer4  # 2048
        
        self.attn1 = AttentionBlock(256, 2048, 128, 8)
        self.attn2 = AttentionBlock(512, 2048, 256, 4)
        self.attn3 = AttentionBlock(1024, 2048, 512, 2)
        self.attn4 = AttentionBlock(2048, 2048, 1024, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        l0 = self.layer1(x)
        l1 = self.layer2(l0)
        l2 = self.layer3(l1)
        l3 = self.layer4(l2)
        
        g = self.pool(l3).flatten(1)
        return torch.cat([
            g,
            self.attn1(l0, l3),
            self.attn2(l1, l3),
            self.attn3(l2, l3),
            self.attn4(l3, l3)
        ], dim=1)

class ResNetAttentionEmbedder:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AttnResNet().to(self.device)
        
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Remove classifier weights if present
            state_dict = {k: v for k, v in state_dict.items() 
                        if not k.startswith('classifier.')}
            
            self.model.load_state_dict(state_dict, strict=False)
            
        self.model.eval()
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        

    def __call__(self, image_path):
        """Process image and return embeddings"""
        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model(tensor).cpu().numpy().squeeze()
        
        # Save embeddings with matching base name
        # base_path = os.path.splitext(image_path)[0]
        # np.save(f"{base_path}.npy", embeddings)
        # print(f"embeddings shape: {embeddings.shape}")
        return embeddings

# if __name__ == "__main__":
#     embedder = ResNetAttentionEmbedder(
#         model_path="src/models/embedding_extractor.pth"
#     )
    
#     embeddings = embedder("/home/sentinel/projects/santanu/front_back_side_view/data/side/image_2.jpg")
    
#     print(f"Embedding shape: {embeddings.shape}")  # Should be (5888,)
