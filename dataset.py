import torchvision
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import json

# Chemins vers les données COCO
data_dir = "D:\\COCO2017"
train_dir = f"{data_dir}/train2017"
val_dir = f"{data_dir}/val2017"
train_ann_file = f"{data_dir}/annotations/instances_train2017.json"
val_ann_file = f"{data_dir}/annotations/instances_val2017.json"

# Transformation pour redimensionner l'image et ajuster les annotations
class ResizeWithAnnotations:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        # Vérification du type d'image
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image, got {type(image)}")

        original_width, original_height = image.size
        # Redimensionnement de l'image
        image = torchvision.transforms.functional.resize(image, self.size)
        new_width, new_height = self.size

        # Ajustement des annotations
        for t in target:
            bbox = t['bbox']
            # Calcul des ratios de redimensionnement
            x_scale = new_width / original_width
            y_scale = new_height / original_height
            
            # Mise à jour des coordonnées de la boîte
            bbox[0] *= x_scale
            bbox[1] *= y_scale
            bbox[2] *= x_scale
            bbox[3] *= y_scale

        return image, target

# Dataset personnalisé avec gestion intégrée du redimensionnement
class CustomCocoDetection(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile, transform=None)
        self.resize_transform = ResizeWithAnnotations((640, 640))
        self.user_transform = transform

    def __getitem__(self, index):
        # Récupération de l'image originale (PIL) et des annotations
        image, target = super().__getitem__(index)
        
        # Application du redimensionnement et ajustement des annotations
        image, target = self.resize_transform(image, target)
        
        # Conversion en Tensor si spécifié
        if self.user_transform:
            image = self.user_transform(image)
            
        return image, target

# Transformation finale (conversion en Tensor)
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # Convertit en Tensor [0, 1]
    torchvision.transforms.Normalize(   # Normalisation ImageNet
        mean=[0.485, 0.456, 0.406],    # Moyenne ImageNet
        std=[0.229, 0.224, 0.225]      # Écart-type ImageNet
    )
])

# Chargement des datasets
train_dataset = CustomCocoDetection(train_dir, train_ann_file, transform=transform)
val_dataset = CustomCocoDetection(val_dir, val_ann_file, transform=transform)

# Création des DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# Chargement des catégories pour l'affichage
with open(train_ann_file, 'r') as f:
    coco_data = json.load(f)
categories = coco_data['categories']
category_id_to_name = {category['id']: category['name'] for category in categories}

# Fonction d'affichage avec les nouvelles coordonnées
def show_image_with_boxes(ax, image_tensor, targets):
    # Conversion du tensor en image PIL
    image = image_tensor.permute(1, 2, 0).numpy()
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

# Visualisation des résultats
if __name__ == "__main__":
    
    # ----------------------------------------------------------------------
    # Test affichage du dataset avec les bounding box associées
    # ----------------------------------------------------------------------
    
    # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    # for i, ax in enumerate(axs.flat):
    #     # Chargement d'un exemple
    #     image_tensor, targets = train_dataset[i+8]  # Changer l'indice pour d'autres exemples
        
    #     # Affichage
    #     show_image_with_boxes(ax, image_tensor, targets)
    #     ax.axis('off')

    # plt.tight_layout()
    # plt.show()

    # ----------------------------------------------------------------------
    # Test de passer les données dans le backbone
    # ----------------------------------------------------------------------
    
    import torch
    from model import ResNet50Backbone
    
    # Vérifier si CUDA est disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Instancier le backbone
    backbone = ResNet50Backbone().to(device)

    # Récupérer un batch de données
    batch = next(iter(train_loader))
    images, targets = batch
    images = images.to(device)

    # Afficher les dimensions d'entrée
    print("\nDimensions d'entrée:")
    print(f"Batch shape: {images.shape}")  # Devrait être [8, 3, 640, 640]

    # Forward pass
    with torch.no_grad():
        features = backbone(images)

    # Afficher les dimensions de sortie
    print("\nDimensions des features maps en sortie:")
    print("C2 (layer1):", features[0].shape)  # [N, 256, 160, 160]
    print("C3 (layer2):", features[1].shape)  # [N, 512, 80, 80]
    print("C4 (layer3):", features[2].shape)  # [N, 1024, 40, 40]
    print("C5 (layer4):", features[3].shape)  # [N, 2048, 20, 20]

    # Vérification mémoire CUDA
    if device.type == 'cuda':
        print("\nUtilisation mémoire CUDA:")
        print(torch.cuda.memory_allocated(device=device) / 1024**3, "GB")