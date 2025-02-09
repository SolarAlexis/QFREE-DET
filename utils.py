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

class PoCooLoss(nn.Module):
    def __init__(self, alpha=0.5, iou_threshold=0.5):
        super(PoCooLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction="none")  # Per-element loss
        self.alpha = alpha  # Contrôle l'importance des petits objets
        self.iou_threshold = iou_threshold

    def forward(self, B_final, final_class_scores, gt_boxes, gt_labels, img_size):
        H, W = img_size
        B, M, _ = B_final.shape
        B, J, _ = gt_boxes.shape
        _, _, num_classes = gt_labels.shape  # Doit être égal à 80

        # === Step 1: Calcul de l'IoU entre B_final et gt_boxes ===
        iou_matrix = compute_iou(B_final, gt_boxes)  # [B, M, J]

        # === Step 2: Association de chaque prédiction à la meilleure boîte ground truth ===
        matched_iou, best_gt_idx = iou_matrix.max(dim=-1)  # [B, M]
        matched_boxes = torch.gather(gt_boxes, 1, best_gt_idx.unsqueeze(-1).expand(-1, -1, 4))  # [B, M, 4]
        matched_labels = torch.gather(gt_labels, 1, best_gt_idx.unsqueeze(-1).expand(-1, -1, num_classes))  # [B, M, 80]

        # === Step 3: Calcul du facteur de pondération selon la taille de la boîte ===
        # Dans le format (x, y, w, h), l'indice 2 est la largeur et l'indice 3 est la hauteur.
        box_widths  = matched_boxes[..., 2]  # Extraire la largeur
        box_heights = matched_boxes[..., 3]  # Extraire la hauteur
        size_factor = (1 - torch.sqrt((box_widths / W) * (box_heights / H))) ** self.alpha  # [B, M]

        # === Step 4: Définition des masques positifs et négatifs en fonction de l'IoU ===
        pos_mask = (matched_iou > self.iou_threshold).float()  # 1 si IoU > 0.5, sinon 0
        neg_mask = (pos_mask == 0).float()  # 1 si IoU < 0.5, sinon 0

        # === Step 5: Calcul de la BCE Loss pour la classification ===
        # On met à zéro les labels des négatifs
        matched_labels = matched_labels * pos_mask.unsqueeze(-1)

        bce_pos = self.bce_loss(final_class_scores, matched_labels)  # [B, M, 80]
        weighted_bce_pos = bce_pos * pos_mask.unsqueeze(-1) * (size_factor.unsqueeze(-1) + 1)  # [B, M, 80]

        bce_neg = self.bce_loss(final_class_scores, torch.zeros_like(final_class_scores))  # BCE avec cible 0
        weighted_bce_neg = (final_class_scores ** 2) * bce_neg * neg_mask.unsqueeze(-1)  # [B, M, 80]

        # === Step 6: Calcul final de la loss ===
        loss = weighted_bce_pos.sum() + weighted_bce_neg.sum()
        return loss / B  # Normalisation par la taille du batch