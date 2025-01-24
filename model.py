import torch
import torch.nn as nn

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
    
