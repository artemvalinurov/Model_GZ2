import torch
import torch.nn as nn
import timm

class CFG:
    backbone = "resnet18"
    p_drop   = 0.3

class GalaxyNet(nn.Module):
    def __init__(self, backbone=CFG.backbone, p_drop=CFG.p_drop):
        super().__init__()
        # Создаем базовый бэкбон ResNet-18 без финального классификатора
        self.trunk = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="avg")
        d = self.trunk.num_features
        
        # Общая полносвязная шейка (neck)
        self.neck = nn.Sequential(
            nn.Linear(d, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), 
            nn.Dropout(p_drop),
        )
        
        # 4 специализированных головы предсказания морфологии
        self.h_type = nn.Linear(256, 3)   # Классификация: гладкая / дисковая / звезда
        self.h_odd  = nn.Linear(256, 4)   # Особенности: нормальная / слияние / возмущенная / прочие
        self.h_edge = nn.Linear(256, 1)   # Сигмоида: P(видна с ребра | диск)
        self.h_bar  = nn.Linear(256, 1)   # Сигмоида: P(есть бар | диск, плашмя)

    def forward(self, x):
        # Если пришел один батч (одиночная картинка), BatchNorm1d требует размерность > 1
        if x.size(0) == 1:
            self.eval() # Принудительно отключаем обновление батч-нормализации
            
        z = self.neck(self.trunk(x))
        return {
            "type": self.h_type(z), 
            "odd": self.h_odd(z),
            "edge": self.h_edge(z).squeeze(-1), 
            "bar": self.h_bar(z).squeeze(-1)
        }
