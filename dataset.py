import torchvision
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
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
            # Copier le dictionnaire d'annotation
            new_t = t.copy()
            bbox = new_t['bbox'].copy()
            
            # Calcul des ratios
            x_scale = new_width / original_width
            y_scale = new_height / original_height
            
            # Mise à l'échelle
            bbox[0] *= x_scale
            bbox[1] *= y_scale
            bbox[2] *= x_scale
            bbox[3] *= y_scale
            
            new_t['bbox'] = bbox
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

# Création des DataLoaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

# Chargement des catégories pour l'affichage
with open(train_ann_file, 'r') as f:
    coco_data = json.load(f)
categories = coco_data['categories']
category_id_to_name = {category['id']: category['name'] for category in categories}

# Fonction d'affichage avec les nouvelles coordonnées
def show_image_with_boxes(ax, image_tensor, targets, normalized=True):
    
    # Dénormalisation si nécessaire
    if normalized:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image_tensor * std + mean  # Dénormalisation
    else:
        image = image_tensor
    
    # Conversion du tensor en image PIL
    image = image.permute(1, 2, 0).numpy()  # Utiliser la variable `image` dénormalisée
    image = np.clip(image, 0, 1)  # S'assurer que les valeurs sont dans [0, 1]
    image = (image * 255).astype(np.uint8)
    ax.imshow(image)

    # Dessin des boîtes redimensionnées
    for target in targets:
        bbox = target['bbox']
        category_id = target['category_id']
        class_name = category_id_to_name.get(category_id, f"Unknown ({category_id})")

        # Création du rectangle
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2],
            bbox[3],
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1], class_name, color='white', backgroundcolor='red', fontsize=8)