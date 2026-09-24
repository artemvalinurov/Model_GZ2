import torch
import torch.nn as nn
import timm

class CFG:
    backbone = "resnet18"
    p_drop   = 0.3

class GalaxyNet(nn.Module):
    def __init__(self, backbone=CFG.backbone, p_drop=CFG.p_drop):
        super().__init__()
        # Создаем ResNet-18 без предсказывающих слоев
        self.trunk = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="avg")
        d = self.trunk.num_features
        
        # Общая шейка признаков
        self.neck = nn.Sequential(
            nn.Linear(d, 256), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(p_drop),
        )
        
        # 4 независимые головы предсказания морфологии под маппинг GZ2
        self.h_type = nn.Linear(256, 3)   # Классификация базового типа: [smooth, disk, star]
        self.h_odd  = nn.Linear(256, 4)   # Специфичные признаки: [normal, merger, disturbed, other]
        self.h_edge = nn.Linear(256, 1)   # Сигмоида: Вероятность "с ребра" (P(edge-on | disk))
        self.h_bar  = nn.Linear(256, 1)   # Сигмоида: Вероятность наличия бара (P(bar | disk, face-on))

    def forward(self, x):
        # Если пришла ровно одна картинка, BatchNorm1d требует отключения обновления статистики
        if x.size(0) == 1:
            self.eval()
        z = self.neck(self.trunk(x))
        return {
            "type": self.h_type(z), 
            "odd": self.h_odd(z),
            "edge": self.h_edge(z).squeeze(-1), 
            "bar": self.h_bar(z).squeeze(-1)
        }
