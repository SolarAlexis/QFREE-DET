import os
import json
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# Importez vos définitions du dataset
from dataset import train_loader, val_loader, train_ann_data  # (train_ann_data contient les annotations COCO)
 
# Importez vos modules de modèle (adaptez les chemins si besoin)
from model import ResNet50Backbone, TransformerEncoder, AFQS, BoxLocatingPart, DeduplicationPart, QFreeDet
from utils import qfreedet_loss

# -----------------------------------------------------------------------------
# Fonction pour préparer les ground truths (boxes et labels) à partir des annotations
# -----------------------------------------------------------------------------
def prepare_ground_truths(targets, num_classes, device, cat_id_to_index):
    """
    Convertit une liste (taille B) de listes d'annotations en tenseurs
    uniformisés par padding.
    
    Args:
        targets (list): Liste (taille B) de listes d'annotations (chaque annotation est un dict contenant au moins "bbox" et "category_id").
        num_classes (int): Par exemple, 80.
        device (torch.device): Le device d'exécution.
        cat_id_to_index (dict): Mapping des category_id (du COCO) vers des indices contigus (0 à num_classes-1).
        
    Returns:
        gt_boxes (Tensor): de forme [B, max_objects, 4].
        gt_labels (Tensor): de forme [B, max_objects, num_classes] en one-hot.
    """
    batch_size = len(targets)
    max_objects = max(len(anns) for anns in targets)
    
    gt_boxes_list = []
    gt_labels_list = []
    for anns in targets:
        num_obj = len(anns)
        if num_obj > 0:
            boxes = torch.stack([torch.tensor(ann["bbox"], dtype=torch.float32, device=device)
                                  for ann in anns], dim=0)
        else:
            boxes = torch.empty((0, 4), device=device)
            
        if num_obj > 0:
            labels = torch.zeros((num_obj, num_classes), device=device)
            for i, ann in enumerate(anns):
                # Conversion via le mapping pour obtenir un indice contigu
                cat = cat_id_to_index[ann["category_id"]]
                labels[i, cat] = 1.0
        else:
            labels = torch.empty((0, num_classes), device=device)
            
        # Padding si nécessaire
        if num_obj < max_objects:
            pad_boxes = torch.zeros((max_objects - num_obj, 4), device=device)
            pad_labels = torch.zeros((max_objects - num_obj, num_classes), device=device)
            boxes = torch.cat([boxes, pad_boxes], dim=0)
            labels = torch.cat([labels, pad_labels], dim=0)
            
        gt_boxes_list.append(boxes)
        gt_labels_list.append(labels)
    
    gt_boxes = torch.stack(gt_boxes_list, dim=0)  # [B, max_objects, 4]
    gt_labels = torch.stack(gt_labels_list, dim=0)  # [B, max_objects, num_classes]
    return gt_boxes, gt_labels

# -----------------------------------------------------------------------------
# Fonction d'entraînement
# -----------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, device, num_epochs):
    model.to(device)
    num_classes = 80
    img_size = 640  # Les images sont redimensionnées à 640x640

    # --- Construction du mapping des catégories ---
    # On charge le fichier d'annotations COCO pour récupérer la liste des catégories
    train_ann_file = os.path.join("D:\\COCO2017", "annotations", "instances_train2017.json")
    with open(train_ann_file, 'r') as f:
        coco_data = json.load(f)
    categories = coco_data['categories']
    # On trie les identifiants (ils ne sont pas forcément contigus dans COCO)
    cat_ids = sorted([cat['id'] for cat in categories])
    cat_id_to_index = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
    
    # --- Initialisation de l'optimizer AdamW ---
    # Taux d'apprentissage : 1e-05 pour le backbone et 1e-04 pour le reste
    backbone_params = list(model.backbone.parameters())
    other_params = [p for name, p in model.named_parameters() if "backbone" not in name]
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": 1e-05, "weight_decay": 0.0001},
        {"params": other_params, "lr": 0.0001, "weight_decay": 0.0001}
    ])
    
    train_losses = []
    val_losses = []
    first_epoch_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # Barre de progression pour l'entraînement
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for images, targets in train_bar:
            images = images.to(device)
            # Préparer les ground truths pour le batch
            gt_boxes, gt_labels = prepare_ground_truths(targets, num_classes, device, cat_id_to_index)
            
            # Passage forward complet : "full" retourne les sorties des branches BLP et DP
            B_blp, class_scores_blp, B_final, final_class_scores = model(images, stage="full")
            loss = qfreedet_loss(B_blp, class_scores_blp, B_final, final_class_scores,
                                 gt_boxes, gt_labels, (img_size, img_size))
            
            optimizer.zero_grad()
            loss.backward()
            # Clipping des gradients (max_norm = 0.1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
            if epoch == 0:
                first_epoch_losses.append(loss.item())
            
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Phase de validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            for images, targets in val_bar:
                images = images.to(device)
                gt_boxes, gt_labels = prepare_ground_truths(targets, num_classes, device, cat_id_to_index)
                B_blp, class_scores_blp, B_final, final_class_scores = model(images, stage="full")
                loss = qfreedet_loss(B_blp, class_scores_blp, B_final, final_class_scores,
                                     gt_boxes, gt_labels, (img_size, img_size))
                running_val_loss += loss.item()
                val_bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if epoch == 0:
            plt.figure(figsize=(8, 5))
            plt.plot(first_epoch_losses, label="Train Loss (1ère époque)")
            plt.xlabel("Itération")
            plt.ylabel("Loss")
            plt.title("Courbe de la loss sur la première époque")
            plt.legend()
            plt.grid(True)
            plt.savefig("first_epoch_loss.png")
            plt.show()
            print("Courbe de la loss de la première époque sauvegardée sous 'first_epoch_loss.png'.")
        
    # Tracé et sauvegarde des courbes de loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curves.png")
    plt.show()
    
    print("Training complete. Loss curves saved as 'loss_curves.png'.")

# -----------------------------------------------------------------------------
# Bloc principal
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialisation des composants du modèle
    backbone = ResNet50Backbone().to(device)
    encoder = TransformerEncoder().to(device)
    afqs = AFQS(threshold=0.5, max_pool_size=100, num_classes=80, feature_dim=256).to(device)
    
    # Paramètres pour BoxLocatingPart (BLP) et DeduplicationPart (DP)
    embed_dim = 256
    D = 256
    H, W = 20, 20
    T1 = 3
    T2 = 3
    lambd = 1
    blp = BoxLocatingPart(embed_dim=embed_dim, H=H, W=W, D=D, T1=T1).to(device)
    dp = DeduplicationPart(embed_dim=embed_dim, H=H, W=W, D=D, T2=T2, lambd=lambd).to(device)
    
    # Création du modèle QFreeDet
    model = QFreeDet(backbone=backbone, afqs=afqs, encoder=encoder, blp=blp, dp=dp).to(device)
    
    # Nombre d'époques d'entraînement
    num_epochs = 2
    train_model(model, train_loader, val_loader, device, num_epochs)