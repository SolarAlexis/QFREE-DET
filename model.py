import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import math

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


class PositionalEncoding2D(nn.Module):
    """Encodage positionnel 2D avec correction de la périodicité"""
    
    def __init__(self, d_model):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model doit être divisible par 4")
        
        self.d_model = d_model
        self.dim = d_model // 4  # 64 pour d_model=256

    def forward(self, B, H, W, device):
        pe = torch.zeros(B, H, W, self.d_model, device=device)
        
        # 🔹 Normalisation correcte du facteur de fréquence
        div_term = torch.exp(
            torch.arange(0, self.dim, device=device, dtype=torch.float32) *
            (-math.log(10000.0) / (self.dim - 1))
        )  

        # Encodage hauteur (y)
        y_pos = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1)  # [H, 1]
        y_enc = torch.cat([torch.sin(y_pos * div_term), torch.cos(y_pos * div_term)], dim=-1)  # [H, 2*dim]
        y_enc = y_enc.unsqueeze(1).expand(-1, W, -1).unsqueeze(0)  # [1, H, W, 2*dim]
        pe[..., :self.d_model//2] = y_enc  # [B, H, W, d_model//2]

        # Encodage largeur (x)
        x_pos = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(1)  # [W, 1]
        x_enc = torch.cat([torch.sin(x_pos * div_term), torch.cos(x_pos * div_term)], dim=-1)  # [W, 2*dim]
        x_enc = x_enc.unsqueeze(0).expand(H, -1, -1).unsqueeze(0)  # [1, H, W, 2*dim]
        pe[..., self.d_model//2:] = x_enc  # [B, H, W, d_model//2]

        return pe

class MultiHeadAttention(nn.Module):
    """Implémentation manuelle de la multi-head attention (batch first).
       Shapes:
         - query, key, value: (B, N, D)
         - retour: (attn_output, attn_weights) avec
             * attn_output: (B, N, D)
             * attn_weights: (B, nhead, N, N) (optionnel si on veut l'extraire)
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = nn.Dropout(dropout)

        assert d_model % nhead == 0, "d_model doit être divisible par nhead"
        self.d_k = d_model // nhead  # dimension de chaque tête

        # Projections linéaires pour Q, K, V et la sortie
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.constant_(self.W_q.bias, 0)
        nn.init.constant_(self.W_k.bias, 0)
        nn.init.constant_(self.W_v.bias, 0)
        nn.init.constant_(self.W_o.bias, 0)

    def forward(self, query, key, value, attn_mask=None):
        """
        Args:
            query, key, value: (B, N, D)
            attn_mask: (B, nhead, N, N) ou broadcastable si besoin (optionnel)
        Returns:
            attn_output: (B, N, D)
            attn_weights: (B, nhead, N, N) si on veut inspecter l'attention
        """
        B, N, _ = query.size()

        # 1. Projection linéaire + découpage en nhead
        Q = self.W_q(query).view(B, N, self.nhead, self.d_k).transpose(1, 2)  # (B, nhead, N, d_k)
        K = self.W_k(key).view(B, -1, self.nhead, self.d_k).transpose(1, 2)   # (B, nhead, N, d_k)
        V = self.W_v(value).view(B, -1, self.nhead, self.d_k).transpose(1, 2) # (B, nhead, N, d_k)

        # 2. Calcul des scores d'attention (scaled dot-product)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, nhead, N, N)
        
        # 3. Application éventuelle d'un masque
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # 4. Softmax sur la dimension des clés
        attn_weights = F.softmax(scores, dim=-1)  # (B, nhead, N, N)
        attn_weights = self.dropout(attn_weights)
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # 5. Calcul de la sortie pondérée
        out = torch.matmul(attn_weights, V)  # (B, nhead, N, d_k)

        # 6. Réassemblage des têtes
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)  # (B, N, D)

        # 7. Projection finale
        attn_output = self.W_o(out)

        return attn_output, attn_weights

class DeformableAttention(nn.Module):
    """Implémentation de l'attention déformable avec prédiction d'offsets.
       Shapes :
         - x : (B, N, D) (entrée sous forme de tokens)
         - retour : (B, N, D) (tokens après attention)
    """
    def __init__(self, d_model=256, nhead=8, num_points=4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_points = num_points
        
        self.offset_pred = nn.Conv2d(d_model, nhead * num_points * 2, kernel_size=3, padding=1)
        self.attn_pred = nn.Conv2d(d_model, nhead * num_points, kernel_size=3, padding=1)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.offset_pred.weight, 0)
        nn.init.constant_(self.offset_pred.bias, 0)
        nn.init.xavier_uniform_(self.attn_pred.weight)
        nn.init.constant_(self.attn_pred.bias, 0)

    def forward(self, x, H, W):
        """
        Args:
            x : (B, N, D)  -> Entrée (tokens encodeur)
            H, W : hauteur et largeur de la feature map d'origine
        Returns:
            out : (B, N, D)  -> Tokens après attention déformable
        """
        B, N, D = x.shape
        x_spatial = x.view(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]
        
        # 1. Prédiction des paramètres
        offsets = self.offset_pred(x_spatial).view(B, self.nhead, self.num_points, 2, H, W)
        attn = F.softmax(self.attn_pred(x_spatial).view(B, self.nhead, self.num_points, H, W), dim=2)
        
        # 2. Calcul des positions
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, H, device=x.device),
            torch.linspace(0.0, 1.0, W, device=x.device),
            indexing='ij'
        )
        ref = torch.stack((ref_x, ref_y), 0)  # [2, H, W]
        ref = ref.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, 2, 1, 1, H, W]
        pos = ref.expand(B, 2, self.nhead, self.num_points, H, W) + offsets.permute(0, 3, 1, 2, 4, 5) * 0.1
        pos = pos.permute(0, 2, 3, 5, 4, 1).reshape(B*self.nhead*self.num_points, H, W, 2)
        
        # 3. Échantillonnage et agrégation
        sampled = F.grid_sample(
            x_spatial.repeat_interleave(self.nhead*self.num_points, 0),
            pos,
            align_corners=False
        ).view(B, self.nhead, self.num_points, D, H, W)
        
        # 4. Combinaison finale corrigée
        out = (sampled * attn.unsqueeze(3)).sum(2)  # [B, nhead, D, H, W]
        out = out.permute(0, 2, 3, 4, 1).reshape(B, D, H*W, self.nhead)  # Nouveau reshape
        out = out.mean(dim=-1).permute(0, 2, 1)  # Fusion des têtes
        
        return self.output_proj(out)

class TransformerEncoderLayer(nn.Module):
    """Couche encodeur avec gestion dynamique du type d'attention"""
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, 
                 dropout=0.1, attention_type='deformable', num_points=4):
        super().__init__()
        self.attention_type = attention_type
        
        if attention_type == 'deformable':
            self.self_attn = DeformableAttention(d_model, nhead, num_points)
        else:
            self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, H=None, W=None):
        # Attention
        if self.attention_type == 'deformable':
            src2 = self.self_attn(src, H, W)
        else:
            src2, _ = self.self_attn(src, src, src)  # (B, N, D)
        
        # Residual + Norm
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        
        return src

class TransformerEncoder(nn.Module):
    """Encodeur avec gestion précise des dimensions spatiales"""
    def __init__(self, d_model=256, nhead=8, num_layers=6, 
                 dim_feedforward=1024, dropout=0.1, 
                 attention_type='deformable', num_points=4):
        super().__init__()
        
        self.projection = nn.Conv2d(2048, d_model, 1)
        self.pos_encoder = PositionalEncoding2D(d_model)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                attention_type=attention_type,
                num_points=num_points
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: [B, 2048, 20, 20] (pour entrée 640x640)
        x = self.projection(x)  # [B, 256, 20, 20]
        B, D, H, W = x.shape
        
        # Encodage positionnel dynamique
        pos = self.pos_encoder(B, H, W, x.device)  # [B, 20, 20, 256]
        
        # Fusion features + position
        x = x.permute(0, 2, 3, 1)  # [B, 20, 20, 256]
        x = x + pos
        x = x.view(B, H*W, D)  # [B, 400, 256]

        # Passage dans les couches
        for layer in self.layers:
            if layer.attention_type == 'deformable':
                x = layer(x, H, W)
            else:
                x = layer(x)
        
        return x  # [B, 400, 256]

class QFreeDet(nn.Module):
    def __init__(self, backbone, afqs, encoder):
        super().__init__()
        self.backbone = backbone    # ResNet50Backbone
        self.encoder = encoder      # Transformer Encoder
        self.afqs = afqs            # AFQS module

    def forward(self, x):
        # 1. Extract multi-scale features
        features = self.backbone(x)
        
        # 2. Use last feature map (c5) for encoding
        c5 = features[-1]  # [B, 2048, H, W]
        
        # 3. Transform to encoder tokens
        encoder_tokens = self.encoder(c5)  # [B, N, D]
        
        # 4. Query selection
        SADQ, selection_mask = self.afqs(encoder_tokens)
        
        return SADQ, selection_mask
