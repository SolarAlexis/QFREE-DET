import torch
import torch.nn as nn

def compute_iou(boxes_preds, boxes_gt):
    """
    Calcule l'Intersection over Union (IoU) entre chaque boîte prédite et toutes les boîtes ground truth.

    Args:
        boxes_preds (Tensor): Boîtes prédites [B, M, 4] sous la forme (x, y, w, h)
        boxes_gt (Tensor): Boîtes de vérité terrain [B, J, 4] sous la forme (x, y, w, h)

    Returns:
        iou (Tensor): Matrice IoU [B, M, J] où chaque prédiction est comparée avec chaque ground truth.
    """
    B, M, _ = boxes_preds.shape
    B, J, _ = boxes_gt.shape

    # Transformation (x, y, w, h) -> (x1, y1, x2, y2) pour les prédictions
    pred_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2  # [B, M, 1]
    pred_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    pred_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    pred_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

    # Transformation (x, y, w, h) -> (x1, y1, x2, y2) pour les ground truth
    gt_x1 = boxes_gt[..., 0:1] - boxes_gt[..., 2:3] / 2  # [B, J, 1]
    gt_y1 = boxes_gt[..., 1:2] - boxes_gt[..., 3:4] / 2
    gt_x2 = boxes_gt[..., 0:1] + boxes_gt[..., 2:3] / 2
    gt_y2 = boxes_gt[..., 1:2] + boxes_gt[..., 3:4] / 2

    # Expansion des dimensions pour comparer toutes les prédictions avec toutes les ground truth
    pred_x1 = pred_x1.expand(-1, -1, J)  # [B, M, J]
    pred_y1 = pred_y1.expand(-1, -1, J)
    pred_x2 = pred_x2.expand(-1, -1, J)
    pred_y2 = pred_y2.expand(-1, -1, J)

    gt_x1 = gt_x1.permute(0, 2, 1)  # [B, 1, J] → [B, J, 1] → [B, J, M] → [B, M, J]
    gt_y1 = gt_y1.permute(0, 2, 1)
    gt_x2 = gt_x2.permute(0, 2, 1)
    gt_y2 = gt_y2.permute(0, 2, 1)

    # Calcul des coordonnées de l'intersection
    inter_x1 = torch.max(pred_x1, gt_x1)
    inter_y1 = torch.max(pred_y1, gt_y1)
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter_y2 = torch.min(pred_y2, gt_y2)

    # Calcul des aires d'intersection et d'union
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    union_area = pred_area + gt_area - inter_area

    # Calcul final de l'IoU (évite division par zéro)
    iou = inter_area / (union_area + 1e-6)  # [B, M, J]

    return iou

# --- Fonctions utilitaires ---

def convert_to_xyxy(boxes):
    """
    Convertit des boîtes du format (x, y, w, h) vers (x1, y1, x2, y2).
    
    Args:
        boxes (Tensor): [B, M, 4] avec (x, y, w, h)
    
    Retourne:
        Tensor: [B, M, 4] avec (x1, y1, x2, y2)
    """
    x = boxes[..., 0]
    y = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)

def compute_giou_loss(pred_boxes, gt_boxes):
    """
    Calcule la perte GIoU entre des boîtes prédites et des boîtes ground truth.
    
    Les boîtes doivent être au format (x1, y1, x2, y2) et de forme [B, M, 4].
    
    Retourne:
        Tensor: perte GIoU par boîte de forme [B, M] (loss = 1 - GIoU)
    """
    # Calcul de l'intersection
    x1 = torch.max(pred_boxes[..., 0], gt_boxes[..., 0])
    y1 = torch.max(pred_boxes[..., 1], gt_boxes[..., 1])
    x2 = torch.min(pred_boxes[..., 2], gt_boxes[..., 2])
    y2 = torch.min(pred_boxes[..., 3], gt_boxes[..., 3])
    
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Aires des boîtes
    area_pred = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    area_gt   = (gt_boxes[..., 2]   - gt_boxes[..., 0])   * (gt_boxes[..., 3]   - gt_boxes[..., 1])
    union = area_pred + area_gt - inter_area + 1e-7
    iou = inter_area / union
    
    # Boîte englobante minimale
    x1_enc = torch.min(pred_boxes[..., 0], gt_boxes[..., 0])
    y1_enc = torch.min(pred_boxes[..., 1], gt_boxes[..., 1])
    x2_enc = torch.max(pred_boxes[..., 2], gt_boxes[..., 2])
    y2_enc = torch.max(pred_boxes[..., 3], gt_boxes[..., 3])
    enc_area = (x2_enc - x1_enc) * (y2_enc - y1_enc) + 1e-7
    
    giou = iou - (enc_area - union) / enc_area
    loss = 1 - giou  # perte à minimiser
    return loss

# --- La PoCoo loss déjà proposée ---

class PoCooLoss(nn.Module):
    def __init__(self, alpha=0.5, iou_threshold=0.5):
        super(PoCooLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction="none")  # loss élément par élément
        self.alpha = alpha  # contrôle l'importance des petits objets
        self.iou_threshold = iou_threshold

    def forward(self, B_final, final_class_scores, gt_boxes, gt_labels, img_size):
        H, W = img_size
        B, M, _ = B_final.shape
        B, J, _ = gt_boxes.shape
        _, _, num_classes = gt_labels.shape  # par exemple, 80 classes

        # 1. Calcul de l'IoU entre chaque boîte prédite et chaque boîte gt
        iou_matrix = compute_iou(B_final, gt_boxes)  # [B, M, J]

        # 2. Association : pour chaque prédiction, on choisit la boîte gt avec IoU maximal
        matched_iou, best_gt_idx = iou_matrix.max(dim=-1)  # [B, M]
        matched_boxes = torch.gather(gt_boxes, 1, best_gt_idx.unsqueeze(-1).expand(-1, -1, 4))  # [B, M, 4]
        matched_labels = torch.gather(gt_labels, 1, best_gt_idx.unsqueeze(-1).expand(-1, -1, num_classes))  # [B, M, num_classes]

        # 3. Calcul du facteur de pondération en fonction de la taille (pour favoriser les petits objets)
        box_widths  = matched_boxes[..., 2]
        box_heights = matched_boxes[..., 3]
        size_factor = (1 - torch.sqrt((box_widths / W) * (box_heights / H))) ** self.alpha  # [B, M]

        # 4. Définition des masques positifs et négatifs (IoU > seuil)
        pos_mask = (matched_iou > self.iou_threshold).float()  # 1 pour IoU > 0.5, 0 sinon
        neg_mask = 1 - pos_mask

        # 5. Calcul de la BCE loss pondérée
        # Pour les positifs, la cible est matched_labels ; pour les négatifs, zéro.
        loss_pos = self.bce_loss(final_class_scores, matched_labels) * pos_mask.unsqueeze(-1) * (size_factor.unsqueeze(-1) + 1)
        loss_neg = self.bce_loss(final_class_scores, torch.zeros_like(final_class_scores)) * (final_class_scores ** 2) * neg_mask.unsqueeze(-1)
        loss = loss_pos.sum() + loss_neg.sum()
        return loss / B  # normalisation par le batch

# --- Perte sur une branche donnée (BLP ou DP) ---

def compute_branch_loss(pred_boxes, pred_scores, gt_boxes, gt_labels, img_size,
                          bce_coeff, l1_coeff, giou_coeff):
    """
    Pour une branche (BLP ou DP), on effectue :
      - une association prédiction <-> gt par IoU,
      - une classification par BCE,
      - une régression L1 sur les coordonnées,
      - une perte GIoU.
    
    Les calculs ne sont effectués que sur les prédictions dites "positives"
    (où l’IoU maximale avec une gt dépasse 0.5).
    
    Args:
        pred_boxes (Tensor): [B, M, 4] (format x,y,w,h)
        pred_scores (Tensor): [B, M, num_classes] (valeurs entre 0 et 1)
        gt_boxes (Tensor): [B, J, 4]
        gt_labels (Tensor): [B, J, num_classes]
        img_size (tuple): (H, W)
        bce_coeff, l1_coeff, giou_coeff (float): coefficients multiplicateurs.
    
    Retourne:
        loss (scalar): perte sur la branche
    """
    B = pred_boxes.shape[0]
    # Association : calcul d'une matrice IoU et sélection du maximum par prédiction
    iou_matrix = compute_iou(pred_boxes, gt_boxes)  # [B, M, J]
    matched_iou, best_gt_idx = iou_matrix.max(dim=-1)  # [B, M]
    # Récupération des gt associées
    matched_gt_boxes = torch.gather(gt_boxes, 1, best_gt_idx.unsqueeze(-1).expand(-1, -1, 4))
    num_classes = gt_labels.shape[-1]
    matched_gt_labels = torch.gather(gt_labels, 1, best_gt_idx.unsqueeze(-1).expand(-1, -1, num_classes))
    
    # On définit le masque positif sur la base d’un seuil IoU de 0.5
    pos_mask = (matched_iou > 0.5).float()  # [B, M]

    # -- Perte de classification BCE --
    bce_loss_fn = nn.BCELoss(reduction='none')
    # Pour les positifs, la cible est matched_gt_labels ; pour les négatifs, zéro.
    loss_cls_pos = bce_loss_fn(pred_scores, matched_gt_labels) * pos_mask.unsqueeze(-1)
    loss_cls_neg = bce_loss_fn(pred_scores, torch.zeros_like(pred_scores)) * (1 - pos_mask).unsqueeze(-1)
    loss_cls = (loss_cls_pos.sum() + loss_cls_neg.sum()) / B

    # -- Perte de régression L1 (uniquement sur les positifs) --
    l1_loss_fn = nn.L1Loss(reduction='none')
    loss_l1 = l1_loss_fn(pred_boxes, matched_gt_boxes).sum(dim=-1)  # [B, M]
    loss_l1 = (loss_l1 * pos_mask).sum() / B

    # -- Perte GIoU (uniquement sur les positifs) --
    pred_boxes_xyxy = convert_to_xyxy(pred_boxes)
    matched_gt_boxes_xyxy = convert_to_xyxy(matched_gt_boxes)
    loss_giou_tensor = compute_giou_loss(pred_boxes_xyxy, matched_gt_boxes_xyxy)  # [B, M]
    loss_giou = (loss_giou_tensor * pos_mask).sum() / B

    total = bce_coeff * loss_cls + l1_coeff * loss_l1 + giou_coeff * loss_giou
    return total

# --- Fonction de perte globale pour QFree-Det ---

def qfreedet_loss(blp_boxes, blp_scores, dp_boxes, dp_scores,
                  gt_boxes, gt_labels, img_size):
    """
    Calcule la loss totale pour un modèle QFree-Det comprenant deux branches :
      - La branche BLP (localisation initiale) et
      - La branche DP (après déduplication)
    On y ajoute également la PoCoo loss pour améliorer la classification des petits objets.
    
    Args:
        blp_boxes (Tensor): [B, M, 4] prédictions BLP (x,y,w,h)
        blp_scores (Tensor): [B, M, num_classes] scores BLP
        dp_boxes (Tensor): [B, M, 4] prédictions DP (x,y,w,h)
        dp_scores (Tensor): [B, M, num_classes] scores DP
        gt_boxes (Tensor): [B, J, 4] ground truth boxes
        gt_labels (Tensor): [B, J, num_classes] ground truth labels
        img_size (tuple): (H, W)
    
    Retourne:
        loss (scalar): loss totale normalisée
    """
    # Coefficients donnés par l'ablation :
    # Pour BLP
    BCE_BLP  = 0.2
    L1_BLP   = 5.0
    GIoU_BLP = 2.0
    # Pour DP
    BCE_DP   = 2.0
    L1_DP    = 2.0
    GIoU_DP  = 2.0

    loss_blp = compute_branch_loss(blp_boxes, blp_scores, gt_boxes, gt_labels, img_size,
                                   BCE_BLP, L1_BLP, GIoU_BLP)
    loss_dp  = compute_branch_loss(dp_boxes, dp_scores, gt_boxes, gt_labels, img_size,
                                   BCE_DP, L1_DP, GIoU_DP)

    # PoCoo loss calculée sur la branche DP (pour améliorer la classification des petits objets)
    pocoo_loss_module = PoCooLoss(alpha=0.5, iou_threshold=0.5)
    loss_pocoo = pocoo_loss_module(dp_boxes, dp_scores, gt_boxes, gt_labels, img_size)

    total_loss = loss_blp + loss_dp + loss_pocoo
    return total_loss