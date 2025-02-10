import torchvision
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
import torch


# Chemins vers les données COCO
data_dir = "D:\\COCO2017"
train_dir = f"{data_dir}/train2017"
val_dir = f"{data_dir}/val2017"
train_ann_file = f"{data_dir}/annotations/instances_train2017.json"
val_ann_file = f"{data_dir}/annotations/instances_val2017.json"

# Transformation pour redimensionner l'image et ajuster les annotations
class ResizeWithAnnotations:
    def __init__(self, size, apply_normalization=True):
        self.size = size
        self.apply_normalization = apply_normalization 
        self.to_tensor = torchvision.transforms.ToTensor()
        
        # Initialiser la normalisation seulement si nécessaire
        if self.apply_normalization:
            self.normalize = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.normalize = None

    def __call__(self, image, target):
        # Créer une copie profonde des annotations
        processed_target = []
        original_width, original_height = image.size
        new_width, new_height = self.size
        
        for t in target:
            new_t = t.copy()
            bbox = new_t['bbox'].copy()  # COCO: [x_min, y_min, w, h]
            
            # Calcul des ratios de mise à l'échelle
            x_scale = new_width / original_width
            y_scale = new_height / original_height
            
            # Mise à l'échelle de la boîte
            bbox[0] *= x_scale  # x_min
            bbox[1] *= y_scale  # y_min
            bbox[2] *= x_scale  # width
            bbox[3] *= y_scale  # height
            
            # Conversion du format COCO (x_min, y_min, w, h) vers (x_center, y_center, w, h)
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            new_bbox = [cx, cy, bbox[2], bbox[3]]
            
            new_t['bbox'] = new_bbox
            processed_target.append(new_t)

        # Redimensionnement de l'image
        image = torchvision.transforms.functional.resize(image, self.size)
        image = self.to_tensor(image)
        
        # Application conditionnelle de la normalisation
        if self.apply_normalization:
            image = self.normalize(image)
        
        return image, processed_target
# Dataset personnalisé avec gestion intégrée du redimensionnement
class FastCocoDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_data, transform=None):
        self.img_dir = img_dir
        self.ann_data = ann_data
        self.transform = transform
        
        # Créer un index image_id -> annotations
        self.img_ann_map = {}
        for ann in ann_data['annotations']:
            self.img_ann_map.setdefault(ann['image_id'], []).append(ann)
        
        self.image_ids = list(sorted(self.img_ann_map.keys()))
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = next((img for img in self.ann_data['images'] if img['id'] == image_id), None)
        
        # Charger l'image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Récupérer les annotations
        anns = self.img_ann_map.get(image_id, [])
        
        # Appliquer les transformations SI ELLES EXISTENT
        if self.transform:
            image, anns = self.transform(image, anns)
        else:
            # Conversion de base si aucune transformation
            image = torchvision.transforms.functional.to_tensor(image)
        
        return image, anns
    
def custom_collate_fn(batch):
    """
    Gère les annotations de taille variable en les retournant sous forme de liste
    """
    # Séparer les images et les annotations
    images = []
    targets = []
    
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    
    # Empiler les images (elles ont toutes la même taille grâce au redimensionnement)
    images = default_collate(images)
    
    return images, targets  # Les annotations restent une liste de dictionnaires


# Transformation finale (conversion en Tensor)
transform = ResizeWithAnnotations((640, 640))

def load_coco_annotations(ann_file):
    from pycocotools.coco import COCO
    coco = COCO(ann_file)
    return {
        "images": coco.dataset['images'],
        "annotations": coco.dataset['annotations'],
        "categories": coco.dataset['categories']
    }

# Charger les annotations une seule fois
train_ann_data = load_coco_annotations(train_ann_file)
val_ann_data = load_coco_annotations(val_ann_file)

# Chargement des datasets
train_dataset = FastCocoDataset(train_dir, train_ann_data, transform=transform)
val_dataset = FastCocoDataset(val_dir, val_ann_data, transform=transform)

# # Création des DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

# Nombre d'images à utiliser pour l'entraînement et la validation
num_train_samples = 100  # Changez selon vos besoins
num_val_samples = 20

# Sélectionner les indices d'échantillons (aléatoires ou premiers N)
train_indices = torch.randperm(len(train_dataset))[:num_train_samples].tolist()
val_indices = torch.randperm(len(val_dataset))[:num_val_samples].tolist()

# Création des sous-ensembles
train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(val_dataset, val_indices)

# Création des DataLoaders avec les sous-ensembles
train_loader = DataLoader(train_subset, batch_size=1, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

# Chargement des catégories pour l'affichage
with open(train_ann_file, 'r') as f:
    coco_data = json.load(f)
categories = coco_data['categories']
category_id_to_name = {category['id']: category['name'] for category in categories}

# Fonction d'affichage avec les nouvelles coordonnées
def show_image_with_boxes(ax, image_tensor, targets, normalized=True, category_mapping=None):
    """
    Affiche une image avec ses bounding boxes.
    Les bounding boxes sont supposées être au format centré (cx, cy, w, h).

    Args:
        ax (matplotlib.axes.Axes): Axe sur lequel dessiner.
        image_tensor (Tensor): Image au format [3, H, W].
        targets (list): Liste d'annotations contenant au moins les clés 'bbox' et 'category_id'.
        normalized (bool): Si True, l'image est dénormalisée avant affichage.
        category_mapping (dict, optionnel): Mapping de category_id vers nom (exemple : category_id_to_name).
    """
    # Si l'image est normalisée, la dénormaliser
    if normalized:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        image_tensor = image_tensor * std + mean

    # Convertir en numpy pour l'affichage (H, W, C)
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    ax.imshow(image_np)

    # Parcourir les annotations et dessiner les bounding boxes
    for target in targets:
        bbox = target['bbox']  # format : [cx, cy, w, h]
        cx, cy, w, h = bbox
        # Conversion en format coin supérieur gauche
        x_min = cx - w / 2
        y_min = cy - h / 2
        # Création d'un rectangle rouge
        rect = plt.Rectangle((x_min, y_min), w, h, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        # Ajout du label si le mapping est fourni
        if category_mapping is not None and 'category_id' in target:
            label = category_mapping.get(target['category_id'], str(target['category_id']))
            ax.text(x_min, y_min, label, color='white', fontsize=10, backgroundcolor='red')