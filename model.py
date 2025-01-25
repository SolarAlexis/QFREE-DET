import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from config import params
class AFQS(nn.Module):
    def __init__(self, threshold=params["AFQS_threshold"], max_pool_size=params["AFQS_max_pool_size"], num_classes=80, feature_dim=256):
        super().__init__()
        self.S = threshold
        self.class_head = nn.Linear(feature_dim, num_classes)
        self.P = max_pool_size

    def forward(self, encoder_tokens):
        B, N, D = encoder_tokens.shape
        class_logits = self.class_head(encoder_tokens)
        class_scores = torch.sigmoid(class_logits).max(dim=-1).values
        selection_mask = class_scores > self.S

        if self.training:
            # Étape 1: Calculer le nombre de requêtes valides par image
            num_valid_list = []
            for b in range(B):
                valid_indices = torch.where(selection_mask[b])[0]
                num_valid_list.append(len(valid_indices))
            
            # Étape 2: Déterminer N_query_b (taille commune du batch)
            N_query_b = min(max(num_valid_list), self.P) if max(num_valid_list) > 0 else self.P

            # Étape 3: Aligner toutes les requêtes à N_query_b
            aligned_queries = []
            for b in range(B):
                valid_indices = torch.where(selection_mask[b])[0]
                num_valid = num_valid_list[b]
                
                # Sélection des tokens valides
                selected = encoder_tokens[b, valid_indices] if num_valid > 0 else torch.zeros((0, D), device=encoder_tokens.device)
                
                padding_needed = N_query_b - num_valid
                available_padding = N - num_valid  # Tokens non sélectionnés disponibles

                if padding_needed > 0:
                    if available_padding > 0:
                        # Sélection des pires tokens non valides
                        worst = class_scores[b].topk(min(padding_needed, available_padding), largest=False).indices
                        padding_tokens = encoder_tokens[b, worst]
                    else:
                        padding_tokens = torch.zeros((0, D), device=encoder_tokens.device)
                    
                    # Compléter avec des zéros si nécessaire
                    remaining_padding = padding_needed - len(padding_tokens)
                    if remaining_padding > 0:
                        zero_pad = torch.zeros((remaining_padding, D), device=encoder_tokens.device)
                        padding_tokens = torch.cat([padding_tokens, zero_pad], dim=0)
                    
                    aligned = torch.cat([selected, padding_tokens], dim=0)
                else:
                    aligned = selected[:N_query_b]  # Troncature (cas rare)
                
                aligned_queries.append(aligned)
            
            SADQ = torch.stack(aligned_queries, dim=0)  # Shape: [B, N_query_b, D]
        
        else:
            # Mode inférence: requêtes variables
            SADQ = [encoder_tokens[b, torch.where(selection_mask[b])[0]] for b in range(B)]
        
        return SADQ, selection_mask
    
class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet50(weights="DEFAULT")  # Poids ImageNet
        
        # Extraction des couches (sans avgpool/fc)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # 256 canaux
        self.layer2 = resnet.layer2  # 512 canaux
        self.layer3 = resnet.layer3  # 1024 canaux
        self.layer4 = resnet.layer4  # 2048 canaux

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)  # Résolution 1/4
        c3 = self.layer2(c2) # Résolution 1/8
        c4 = self.layer3(c3) # Résolution 1/16
        c5 = self.layer4(c4) # Résolution 1/32
        return [c2, c3, c4, c5]  # Features multi-échelles
