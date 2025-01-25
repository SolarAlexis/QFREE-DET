import torchvision
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import json

# Chemin vers le dossier COCO
data_dir = "D:\\COCO2017"
train_dir = f"{data_dir}/train2017"
val_dir = f"{data_dir}/val2017"
train_ann_file = f"{data_dir}/annotations/instances_train2017.json"
val_ann_file = f"{data_dir}/annotations/instances_val2017.json"

# Transformation des images (ajustez selon vos besoins)
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((800, 1333))  # Taille d'image standard pour COCO
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

def show_image_with_boxes(image, targets):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for target in targets:
        bbox = target['bbox']
        category_id = target['category_id']
        class_name = category_id_to_name.get(category_id, f"Unknown ({category_id})")
        x, y, width, height = bbox
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y, class_name, color='white', backgroundcolor='red', fontsize=8)

    plt.show()

if __name__ == "__main__":
    # Charger une image et ses annotations
    image, targets = train_dataset[0]  # Vous pouvez changer l'indice pour une autre image

    # Convertir l'image tensor en image PIL pour l'affichage
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)

    # Afficher l'image avec les boîtes englobantes et les noms de classes
    show_image_with_boxes(image, targets)