import os
from PIL import Image
import copy

import torch
import matplotlib.pyplot as plt

from model import AFQS, ResNet50Backbone, PositionalEncoding2D, DeformableAttention, TransformerEncoder, QFreeDet
from dataset import train_dataset, train_loader, train_ann_data, transform, train_dir, show_image_with_boxes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_afqs():
    # Paramètres communs
    feature_dim = 256

    # ----------------------------------------------------------------------
    # Cas 1: Entraînement - Max des requêtes valides (50) < max_pool_size (100)
    # Résultat: N_query_b = 50
    # ----------------------------------------------------------------------
    model = AFQS(threshold=0.5, max_pool_size=100).to(device)
    model.train()

    # Contrôler le class_head pour que les scores soient prédictibles
    with torch.no_grad():
        model.class_head.weight.data.zero_()  # Désactiver tous les poids
        model.class_head.bias.data.zero_()    # Désactiver le biais
        model.class_head.weight.data[0, :] = 1.0  # Classe 0 = somme des features

    # Créer un batch avec [30, 50, 10] requêtes valides
    encoder_tokens = torch.zeros(3, 200, feature_dim, device=device)  # Initialiser à zéro
    with torch.no_grad():
        # Image 0: 30 tokens valides, reste invalides
        encoder_tokens[0, :30] = 10.0   # Scores élevés (classe 0 = 10.0 * 256 = 2560)
        encoder_tokens[0, 30:] = -10.0  # Scores bas (classe 0 = -10.0 * 256 = -2560)
        
        # Image 1: 50 tokens valides, reste invalides
        encoder_tokens[1, :50] = 10.0
        encoder_tokens[1, 50:] = -10.0
        
        # Image 2: 10 tokens valides, reste invalides
        encoder_tokens[2, :10] = 10.0
        encoder_tokens[2, 10:] = -10.0

    SADQ, _ = model(encoder_tokens)
    assert SADQ.shape[1] == 50, f"Cas 1 échoué: {SADQ.shape} != (3, 50, 256)"
    print("Cas 1 (Max valide < max_pool) réussi ✅")

    # ----------------------------------------------------------------------
    # Cas 2: Entraînement - Max des requêtes valides (200) > max_pool_size (100)
    # Résultat: N_query_b = 100
    # ----------------------------------------------------------------------
    model = AFQS(threshold=0.5, max_pool_size=100).to(device)
    model.train()

    encoder_tokens = torch.randn(2, 300, feature_dim, device=device)
    with torch.no_grad():
        encoder_tokens[0, :200] += 10.0  # 200 valides (dépassement)
        encoder_tokens[1, :50] += 10.0   # 50 valides

    SADQ, _ = model(encoder_tokens)
    assert SADQ.shape[1] == 100, f"Cas 2 échoué: {SADQ.shape} != (2, 100, 256)"
    print("Cas 2 (Max valide > max_pool) réussi ✅")

    # ----------------------------------------------------------------------
    # Cas 3: Entraînement - Aucun token valide (remplissage avec des zéros)
    # Résultat: N_query_b = max_pool_size
    # ----------------------------------------------------------------------
    model = AFQS(threshold=1.0, max_pool_size=100).to(device)
    model.train()

    # Forcer tous les scores à être < 1.0
    encoder_tokens = torch.randn(2, 50, feature_dim, device=device) * 0.1  # Scores bas

    SADQ, _ = model(encoder_tokens)
    assert SADQ.shape == (2, 100, feature_dim), f"Échec: {SADQ.shape} != (2, 100, 256)"
    print("Cas 3 (Aucun token valide) réussi ✅")

    # ----------------------------------------------------------------------
    # Cas 4: Inférence - Requêtes variables
    # Résultat: List[Tensor] de tailles différentes
    # ----------------------------------------------------------------------
    model = AFQS(threshold=0.5).to(device)
    model.eval()

    # Contrôler le class_head pour des scores prédictibles
    with torch.no_grad():
        model.class_head.weight.data.zero_()
        model.class_head.bias.data.zero_()
        model.class_head.weight.data[0, :] = 1.0  # Classe 0 = somme des features

    encoder_tokens = torch.zeros(2, 100, feature_dim, device=device)
    with torch.no_grad():
        encoder_tokens[0, :30] = 10.0  # Scores élevés
        encoder_tokens[0, 30:] = -10.0 # Scores bas
        encoder_tokens[1, :60] = 10.0
        encoder_tokens[1, 60:] = -10.0

    SADQ, selection_mask = model(encoder_tokens)
    assert isinstance(SADQ, list), "Erreur: SADQ doit être une liste en inférence"
    assert SADQ[0].shape == (30, feature_dim), f"Image 0: {SADQ[0].shape} != (30, 256)"
    assert SADQ[1].shape == (60, feature_dim), f"Image 1: {SADQ[1].shape} != (60, 256)"
    print("Cas 4 (Inférence variable) réussi ✅")
    
    # ----------------------------------------------------------------------
    # Cas 5: Entraînement - Nombre valide (150) > max_pool_size (100)
    # Résultat: Troncature à P=100 et remplissage correct
    # ----------------------------------------------------------------------
    model = AFQS(threshold=0.5, max_pool_size=100).to(device)
    model.train()

    # Créer un batch avec [150, 50] requêtes valides
    encoder_tokens = torch.zeros(2, 200, feature_dim, device=device)
    with torch.no_grad():
        # Image 0: 150 tokens valides
        encoder_tokens[0, :150] = 10.0  # Top 100 seront gardés
        encoder_tokens[0, 150:] = -10.0
        
        # Image 1: 50 tokens valides
        encoder_tokens[1, :50] = 10.0
        encoder_tokens[1, 50:] = -10.0

    SADQ, _ = model(encoder_tokens)
    
    # Vérifier la forme
    assert SADQ.shape == (2, 100, feature_dim), f"Cas 5 échoué: {SADQ.shape} != (2, 100, 256)"
    
    # Vérifier la troncature pour l'image 0 (doit garder les 100 premiers des 150 valides)
    assert torch.all(SADQ[0] == 10.0), "Erreur troncature top tokens valides"
    
    # Vérifier le remplissage pour l'image 1 (50 valides + 50 pires non-valides)
    assert torch.all(SADQ[1, :50] == 10.0), "Erreur partie valide image 1"
    assert torch.all(SADQ[1, 50:] == -10.0), "Erreur remplissage image 1"
    
    print("Cas 5 (Troncature valide > P) réussi ✅")
    
    # ----------------------------------------------------------------------
    # Cas 6: Entraînement - Un mix exact de P et sous-P dans le même batch
    # Vérifie le cas où une image a exactement P tokens valides
    # et l'autre en a moins, nécessitant un padding précis
    # ----------------------------------------------------------------------
    model = AFQS(threshold=0.5, max_pool_size=100).to(device)
    model.train()

    # Créer un batch avec [100, 30] requêtes valides
    encoder_tokens = torch.zeros(2, 150, feature_dim, device=device)
    with torch.no_grad():
        # Image 0: 100 tokens valides (exactement P)
        encoder_tokens[0, :100] = 10.0
        encoder_tokens[0, 100:] = -10.0
        
        # Image 1: 30 tokens valides
        encoder_tokens[1, :30] = 10.0
        encoder_tokens[1, 30:] = -10.0

    SADQ, _ = model(encoder_tokens)
    
    # Vérifier la forme
    assert SADQ.shape == (2, 100, feature_dim), f"Cas 6 échoué: {SADQ.shape} != (2, 100, 256)"
    
    # Vérifier l'image 0 (exactement P tokens valides, pas de padding)
    assert torch.all(SADQ[0] == 10.0), "Erreur: Image 0 devrait avoir 100 tokens valides exactement"
    
    # Vérifier l'image 1 (30 valides + 70 pires non-valides)
    assert torch.all(SADQ[1, :30] == 10.0), "Erreur partie valide image 1"
    assert torch.all(SADQ[1, 30:100] == -10.0), "Erreur padding image 1"
    
    print("Cas 6 (Mix exact P/sous-P) réussi ✅")

    # ----------------------------------------------------------------------
    # Cas 7: Vérification du flux de gradient complet avec classification loss
    # ----------------------------------------------------------------------
    model = AFQS(threshold=0.3, max_pool_size=50).to(device)
    model.train()

    # Créer des données avec gradient et étiquettes factices
    encoder_tokens = torch.randn(2, 100, feature_dim, device=device, requires_grad=True)
    dummy_labels = torch.randint(0, 80, (2, 100), device=device)  # Étiquettes aléatoires

    # Forward pass
    SADQ, _ = model(encoder_tokens)
    class_logits = model.class_head(encoder_tokens)  # [B, N, num_classes]

    # Calculer une loss combinée : somme des SADQ + perte de classification
    loss_sadq = SADQ.sum() * 0.1
    loss_class = torch.nn.functional.cross_entropy(
        class_logits.view(-1, model.class_head.out_features), 
        dummy_labels.view(-1),
        ignore_index=-1  # Ignorer les tokens non valides si nécessaire
    )
    total_loss = loss_sadq + loss_class

    # Backward pass
    total_loss.backward()

    # Vérifier les gradients
    assert encoder_tokens.grad is not None, "Gradient manquant pour encoder_tokens"
    assert model.class_head.weight.grad is not None, "Gradient manquant pour class_head.weight"
    assert model.class_head.bias.grad is not None, "Gradient manquant pour class_head.bias"

    print("Cas 7 (Flux de gradient complet) réussi ✅")

def test_dataset_and_backbone():
    # ----------------------------------------------------------------------
    # Test 1: Vérification du chargement des données et des transformations
    # ----------------------------------------------------------------------
    # Récupérer un échantillon du dataset
    image_tensor, targets = train_dataset[0]
    
    # Vérifier les dimensions de l'image
    assert image_tensor.shape == (3, 640, 640), f"Shape incorrect: {image_tensor.shape}"
    
    # Vérifier la présence d'annotations
    assert len(targets) > 0, "Aucune annotation trouvée"
    
    # Vérifier le format des annotations
    for ann in targets:
        assert 'bbox' in ann, "Clé 'bbox' manquante"
        assert 'category_id' in ann, "Clé 'category_id' manquante"
        
    print("Test 1 (Chargement données) réussi ✅")

    # ----------------------------------------------------------------------
    # Test 2: Vérification du DataLoader et collate_fn
    # ----------------------------------------------------------------------
    batch = next(iter(train_loader))
    images, targets = batch
    
    # Vérifier le format du batch
    assert images.shape == (8, 3, 640, 640), f"Shape batch incorrect: {images.shape}"
    assert len(targets) == 8, f"Nombre d'annotations incorrect: {len(targets)}"
    
    # Vérifier le type des annotations
    assert isinstance(targets, list), "Les annotations doivent être une liste"
    assert all(isinstance(t, list) for t in targets), "Chaque élément doit être une liste d'annotations"
    
    print("Test 2 (DataLoader) réussi ✅")

    # ----------------------------------------------------------------------
    # Test 3: Vérification des sorties du backbone
    # ----------------------------------------------------------------------
    backbone = ResNet50Backbone().to(device)
    images = images.to(device)
    
    with torch.no_grad():
        features = backbone(images)
    
    # Vérifier les dimensions des features maps
    assert features[0].shape == (8, 256, 160, 160), f"C2 shape incorrect: {features[0].shape}"
    assert features[1].shape == (8, 512, 80, 80), f"C3 shape incorrect: {features[1].shape}"
    assert features[2].shape == (8, 1024, 40, 40), f"C4 shape incorrect: {features[2].shape}"
    assert features[3].shape == (8, 2048, 20, 20), f"C5 shape incorrect: {features[3].shape}"
    
    print("Test 3 (Backbone outputs) réussi ✅")

    # ----------------------------------------------------------------------
    # Test 4: Vérification du redimensionnement des bounding boxes (CORRIGÉ)
    # ----------------------------------------------------------------------
    # Trouver une image avec annotations
    sample_idx = next((i for i, (_, t) in enumerate(train_dataset) if len(t) > 0), None)
    assert sample_idx is not None, "Aucune image annotée trouvée"
    
    # Récupérer les données originales
    image_id = train_dataset.image_ids[sample_idx]
    image_info = next(img for img in train_ann_data['images'] if img['id'] == image_id)
    original_anns = [copy.deepcopy(ann) for ann in train_ann_data['annotations'] if ann['image_id'] == image_id]
    
    # Charger l'image originale
    original_image = Image.open(os.path.join(train_dir, image_info['file_name'])).convert('RGB')
    
    # Appliquer la transformation avec copie
    transformed_image, transformed_anns = transform(copy.deepcopy(original_image), copy.deepcopy(original_anns))
    
    # Vérifier une annotation
    original_bbox = original_anns[0]['bbox']
    transformed_bbox = transformed_anns[0]['bbox']
    
    # Calcul des ratios théoriques
    x_scale = 640 / original_image.width
    y_scale = 640 / original_image.height
    
    # Vérification avec marge d'erreur
    assert abs(transformed_bbox[1] - original_bbox[1] * y_scale) < 1e-4, (
        f"Erreur Y: {transformed_bbox[1]} vs {original_bbox[1] * y_scale} "
        f"(Original H: {original_image.height}px)"
    )
    
    print("Test 4 (BBox scaling) réussi ✅")

    # ----------------------------------------------------------------------
    # Test 5: Vérification de la normalisation des images
    # ----------------------------------------------------------------------
    # 1. Créer une image artificielle rouge uni
    fake_image = Image.new('RGB', (640, 640), color=(255, 0, 0))
    
    # 2. Appliquer la transformation complète
    tensor_img, _ = transform(fake_image, [])
    
    # 3. Calculer les canaux normalisés
    red_channel = tensor_img[0].mean().item()
    green_channel = tensor_img[1].mean().item()
    blue_channel = tensor_img[2].mean().item()
    
    # 4. Valeurs théoriques attendues pour du rouge pur
    expected_red = (255/255 - 0.485)/0.229    # ≈ (1.0 - 0.485)/0.229 ≈ 2.2489
    expected_green = (0/255 - 0.456)/0.224    # ≈ (-0.456)/0.224 ≈ -2.0357
    expected_blue = (0/255 - 0.406)/0.225     # ≈ (-0.406)/0.225 ≈ -1.8044

    # 5. Vérification avec tolérance
    assert abs(red_channel - expected_red) < 0.01, f"R: {red_channel:.4f} vs {expected_red:.4f}"
    assert abs(green_channel - expected_green) < 0.01, f"G: {green_channel:.4f} vs {expected_green:.4f}"
    assert abs(blue_channel - expected_blue) < 0.01, f"B: {blue_channel:.4f} vs {expected_blue:.4f}"
    
    print("Test 5 (Normalisation) réussi ✅")

def test_visualization():
    # ----------------------------------------------------------------------
    # Test 6: Vérification de la visualisation des images et des bounding boxes
    # ----------------------------------------------------------------------
    # Trouver 8 images avec des annotations
    sample_indices = []
    for i in range(1, len(train_dataset)):
        _, targets = train_dataset[i]
        if len(targets) > 0:
            sample_indices.append(i)
            if len(sample_indices) == 8:
                break
    assert len(sample_indices) == 8, "Pas assez d'images avec annotations trouvées"
    
    # Créer une figure pour l'affichage (2 lignes, 4 colonnes)
    fig, axs = plt.subplots(2, 4, figsize=(14, 7))  # Taille ajustée pour 8 images
    
    # Afficher chaque image avec ses bounding boxes
    for i, ax in enumerate(axs.flat):
        # Charger l'image et ses annotations
        image_tensor, targets = train_dataset[sample_indices[i]]
        
        # Vérifier les dimensions de l'image
        assert image_tensor.shape == (3, 640, 640), f"Shape incorrect: {image_tensor.shape}"
        
        # Vérifier la présence d'annotations
        assert len(targets) > 0, "Aucune annotation trouvée pour cette image"
        
        # Afficher l'image avec les bounding boxes
        show_image_with_boxes(ax, image_tensor, targets, normalized=True)
        ax.axis('off')
        
        # Vérifier que l'image est correctement dénormalisée
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        denormalized_image = image_tensor * std + mean
        
        # Vérifier que les valeurs sont dans [0, 1] après dénormalisation
        assert torch.all(denormalized_image >= 0).item(), "Erreur de dénormalisation : valeurs négatives"
        assert torch.all(denormalized_image <= 1).item(), "Erreur de dénormalisation : valeurs > 1"
    
    # Afficher la figure
    plt.tight_layout()
    plt.show()
    
    print("Test 6 (Visualisation) réussi ✅")

def test_positional_encoding():
    """Teste l'encodage positionnel 2D"""
    # 1. Initialisation
    d_model = 256
    pe = PositionalEncoding2D(d_model)
    B, H, W = 2, 20, 20
    
    # 2. Exécution
    encoding = pe(B, H, W, torch.device('cpu'))
    
    # 3. Vérifications
    assert encoding.shape == (B, H, W, d_model), f"Mauvaise shape: {encoding.shape}"
    assert not torch.allclose(encoding[0,0,0], encoding[0,1,0]), "Encodage vertical manquant"
    assert not torch.allclose(encoding[0,0,0], encoding[0,0,1]), "Encodage horizontal manquant"
    print("Test PositionalEncoding2D réussi ✅")

def test_deformable_attention():
    """Teste le module d'attention déformable avec dimensions corrigées"""
    B, H, W, D = 2, 20, 20, 256
    layer = DeformableAttention(D, nhead=8, num_points=4)
    x = torch.randn(B, H*W, D)
    
    output = layer(x, H, W)
    
    assert output.shape == (B, H*W, D), f"Shape sortie: {output.shape}"
    print("Test DeformableAttention réussi ✅")
    
def test_transformer_encoder():
    """Teste l'encodeur Transformer complet"""
    # 1. Configuration
    encoder = TransformerEncoder(
        d_model=256,
        nhead=8,
        num_layers=2,
        attention_type='deformable',
        num_points=4
    )
    
    # 2. Données test (sortie backbone)
    c5 = torch.randn(2, 2048, 20, 20)  # Shape typique pour entrée 640x640
    
    # 3. Forward pass
    encoder_tokens = encoder(c5)
    
    # 4. Vérifications
    assert encoder_tokens.shape == (2, 400, 256), f"Shape tokens incorrect: {encoder_tokens.shape}"
    print("Test TransformerEncoder réussi ✅")

def test_full_model():
    """Test d'intégration complet"""
    # 1. Initialisation avec max_pool_size=100
    backbone = ResNet50Backbone()
    encoder = TransformerEncoder()
    afqs = AFQS(max_pool_size=100) 
    model = QFreeDet(backbone, afqs, encoder)
    
    # 2. Données test
    x = torch.randn(2, 3, 640, 640)
    
    # 3. Forward pass
    sadq, mask = model(x)
    
    # 4. Vérifications
    assert sadq.shape == (2, 100, 256), f"Shape SADQ incorrect: {sadq.shape}"
    assert mask.shape == (2, 400), f"Shape mask incorrect: {mask.shape}"
    print("Test intégration complet réussi ✅")

if __name__ == "__main__":
    
    test_positional_encoding()
    test_deformable_attention()
    test_transformer_encoder()
    test_full_model()
    
    # test_afqs()
    # test_dataset_and_backbone()
    # test_visualization()