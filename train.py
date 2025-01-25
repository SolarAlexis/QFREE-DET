import torchvision
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import json
import torch

# Chemin vers le dossier COCO
data_dir = "D:\\COCO2017"
train_dir = f"{data_dir}/train2017"
val_dir = f"{data_dir}/val2017"
train_ann_file = f"{data_dir}/annotations/instances_train2017.json"
val_ann_file = f"{data_dir}/annotations/instances_val2017.json"

# Transformation des images pour ResNet-50
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224))  # Taille d'image attendue par ResNet-50
])

# Chargement des datasets
train_dataset = CocoDetection(root=train_dir, annFile=train_ann_file, transform=transform)
val_dataset = CocoDetection(root=val_dir, annFile=val_ann_file, transform=transform)

# Création des DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

with open(train_ann_file, 'r') as f:
    coco_data = json.load(f)
categories = coco_data['categories']
category_id_to_name = {category['id']: category['name'] for category in categories}

def show_image_with_boxes(image, targets, original_size, resized_size):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Calculer les facteurs d'échelle
    scale_x = resized_size[0] / original_size[0]
    scale_y = resized_size[1] / original_size[1]

    for target in targets:
        bbox = target['bbox']
        category_id = target['category_id']
        class_name = category_id_to_name.get(category_id, f"Unknown ({category_id})")

        # Ajuster les coordonnées de la boîte englobante
        x, y, width, height = bbox
        x = x * scale_x
        y = y * scale_y
        width = width * scale_x
        height = height * scale_y

        # Dessiner la boîte englobante
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y, class_name, color='white', backgroundcolor='red', fontsize=8)

    plt.show()

if __name__ == "__main__":
    # Charger une image et ses annotations
    image, targets = train_dataset[0]  # Vous pouvez changer l'indice pour une autre image

    # Convertir l'image tensor en image PIL pour l'affichage
    image = image.permute(1, 2, 0).numpy()  # Convertir en tableau NumPy
    image = (image * 255).astype(np.uint8)  # Multiplier par 255 et convertir en uint8
    image_pil = Image.fromarray(image)  # Convertir en image PIL

    # Obtenir la taille originale de l'image
    original_size = train_dataset.coco.loadImgs(train_dataset.ids[0])[0]['width'], train_dataset.coco.loadImgs(train_dataset.ids[0])[0]['height']

    # Taille de l'image redimensionnée
    resized_size = (224, 224)  # Taille attendue par ResNet-50

    # Afficher l'image avec les boîtes englobantes et les noms de classes
    show_image_with_boxes(image_pil, targets, original_size, resized_size)

    # Charger ResNet-50
    resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)  # Utiliser les poids par défaut
    resnet50.eval()  # Passer en mode évaluation

    # Préparer l'image pour ResNet-50
    image_tensor = transform(image_pil)  # Appliquer la transformation
    image_tensor = image_tensor.unsqueeze(0)  # Ajouter une dimension de batch

    # Passer l'image à ResNet-50
    with torch.no_grad():
        features = resnet50(image_tensor)

    print("Features shape:", features.shape)  # Afficher la forme des caractéristiques extraites