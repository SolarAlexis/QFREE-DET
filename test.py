import os
from PIL import Image
import copy

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import AFQS, ResNet50Backbone, PositionalEncoding2D, DeformableAttention, TransformerEncoder, MultiHeadAttention, QFreeDet, BoxLocatingPart, DeduplicationPart
from dataset import train_dataset, train_loader, train_ann_data, transform, train_dir, show_image_with_boxes
from utils import compute_iou, PoCooLoss

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
    # 1) Création de l'instance
    d_model = 256
    pe = PositionalEncoding2D(d_model)
    B, H, W = 2, 10, 10  # batch=2, hauteur=10, largeur=10
    
    # 2) Passage forward
    encoding = pe(B, H, W, device=torch.device('cpu'))
    
    # 3) Vérification de la forme
    assert encoding.shape == (B, H, W, d_model), (
        f"Forme attendue: {(B, H, W, d_model)}, forme obtenue: {encoding.shape}"
    )
    
    # 4) Vérification que l'encodage varie selon la hauteur et la largeur
    # Vérif. variation verticale
    assert not torch.allclose(encoding[0, 0, 0], encoding[0, 1, 0]), (
        "Les encodages à 2 hauteurs différentes sont identiques, "
        "l'encodage vertical ne varie pas."
    )
    # Vérif. variation horizontale
    assert not torch.allclose(encoding[0, 0, 0], encoding[0, 0, 1]), (
        "Les encodages à 2 largeurs différentes sont identiques, "
        "l'encodage horizontal ne varie pas."
    )

    # 5) Vérification sur 2 positions aléatoires
    # (pour s'assurer que l'encodage n'est pas uniforme ou constant)
    pos1 = (0, 2, 3)  # Batch=0, Y=2, X=3
    pos2 = (0, 5, 7)  # Batch=0, Y=5, X=7
    assert not torch.allclose(encoding[pos1], encoding[pos2]), (
        f"Les encodages aux positions {pos1} et {pos2} sont identiques, "
        "alors qu'ils devraient différer."
    )
    
    # 6) Cas limite : 1×1 (vérifier que la forme et l'appel fonctionnent)
    encoding_1x1 = pe(1, 1, 1, device=torch.device('cpu'))
    assert encoding_1x1.shape == (1, 1, 1, d_model), (
        f"Forme attendue pour (1,1): {(1, 1, 1, d_model)}, "
        f"forme obtenue: {encoding_1x1.shape}"
    )
    
    print("Test positional_encoding réussi ✅")

def test_multihead_attention():
    """Teste le module MultiHeadAttention sur différentes dimensions et vérifie
       - la forme de la sortie,
       - la normalisation des poids d'attention,
       - la rétropropagation,
       - éventuellement le fonctionnement d'un masque.
    """
    d_model = 256
    nhead = 8
    layer = MultiHeadAttention(d_model, nhead=nhead, dropout=0.1)

    # On teste plusieurs combinaisons (batch_size, seq_len)
    for (B, N) in [(2, 10), (4, 20), (1, 5)]:
        # 1. Création de tenseurs random
        query = torch.randn(B, N, d_model, requires_grad=True)
        key   = torch.randn(B, N, d_model, requires_grad=True)
        value = torch.randn(B, N, d_model, requires_grad=True)

        # 2. Forward pass (sans masque)
        attn_output, attn_weights = layer(query, key, value)

        # 3. Vérification des formes
        assert attn_output.shape == (B, N, d_model), (
            f"Échec shape attn_output: {attn_output.shape}, attendu={(B, N, d_model)}"
        )
        assert attn_weights.shape == (B, nhead, N, N), (
            f"Échec shape attn_weights: {attn_weights.shape}, attendu={(B, nhead, N, N)}"
        )

        # 4. Vérification que la somme des poids d'attention = 1 (axe -1)
        with torch.no_grad():
            somme = attn_weights.sum(dim=-1)  # (B, nhead, N)
            ones = torch.ones_like(somme)
            assert torch.all(torch.isclose(somme, ones, atol=1e-3)), (
                f"Erreur : Différence max = {torch.abs(somme - ones).max().item()}"
            )

        # 5. Vérification de la rétropropagation (aucune erreur ne doit être levée)
        loss = attn_output.mean()
        loss.backward()

    # 6. Test optionnel avec un masque (par ex. masque triangulaire causal)
    #    On crée un masque binaire de taille (N, N) qu'on broadcast sur (B, nhead, N, N).
    B, N = 2, 5
    causal_mask = torch.tril(torch.ones(N, N, dtype=torch.bool))  # bas-triangulaire
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, nhead, N, N)  # (B, nhead, N, N)

    query = torch.randn(B, N, d_model, requires_grad=True)
    key   = torch.randn(B, N, d_model, requires_grad=True)
    value = torch.randn(B, N, d_model, requires_grad=True)

    attn_output, attn_weights = layer(query, key, value, attn_mask=causal_mask)
    assert attn_output.shape == (B, N, d_model), "Erreur de shape avec masque causal."

    loss = attn_output.mean()
    loss.backward()  # test de backprop

    print("Test MultiHeadAttention réussi ✅")

def test_deformable_attention():
    """Teste le module d'attention déformable avec dimensions corrigées + validations."""
    
    # 1. Création du module
    d_model = 256
    nhead = 8
    num_points = 4
    layer = DeformableAttention(d_model, nhead=nhead, num_points=num_points)

    # 2. Test sur différentes tailles de feature map
    for (B, H, W) in [(2, 20, 20), (2, 10, 25), (1, 5, 5)]:
        N = H * W  # Nombre total de pixels
        x = torch.randn(B, N, d_model, requires_grad=True)  # On autorise la backprop

        # 3. Forward pass
        output = layer(x, H, W)

        # 4. Vérification de la forme
        assert output.shape == (B, N, d_model), (
            f"Échec shape: sortie={output.shape} attendu={(B, N, d_model)} "
            f"pour B={B}, H={H}, W={W}."
        )

        # 5. Vérification des offsets
        with torch.no_grad():
            x_spatial = x.view(B, H, W, d_model).permute(0, 3, 1, 2)  # [B, D, H, W]
            offsets = layer.offset_pred(x_spatial).view(B, nhead, num_points, 2, H, W)
            
            assert offsets.abs().max() < 1.0, "Offsets trop grands, peut être instable."

        # 6. Vérification de la normalisation des poids d'attention
        with torch.no_grad():
            attn = layer.attn_pred(x_spatial).view(B, nhead, num_points, H, W)
            attn_softmax = F.softmax(attn, dim=2)
            assert torch.allclose(attn_softmax.sum(dim=2), torch.ones_like(attn_softmax.sum(dim=2)), atol=1e-3), (
                "Les poids d'attention ne sont pas normalisés (softmax incorrect)."
            )

        # 7. Test de la rétropropagation
        loss = output.mean()
        loss.backward()  # Vérifie qu'aucune erreur n'est levée
        
    print("Test DeformableAttention réussi ✅")
    
def test_transformer_encoder():
    """Teste l'encodeur Transformer complet, avec plusieurs configs:
       - Différents batch sizes
       - Différentes résolutions
       - Deux types d'attention (déformable / multihead)
       - Verification de la forme de sortie et de la backprop
    """
    # On veut tester 2 types d'attention
    attention_types = ['deformable', 'multihead']
    # On veut tester plusieurs (B, H, W)
    test_configs = [
        (1, 5, 5),
        (2, 20, 20),
        (4, 10, 25),
    ]
    
    for attn_type in attention_types:
        # Création d'un encodeur avec 2 couches
        encoder = TransformerEncoder(
            d_model=256,
            nhead=8,
            num_layers=2,
            dim_feedforward=1024,
            dropout=0.1,
            attention_type=attn_type,
            num_points=4
        )

        # Test sur plusieurs batch / résolutions
        for (B, H, W) in test_configs:
            
            # Simule une sortie backbone (ex: ResNet)
            # 2048 channels correspond à la sortie "C5" dans ResNet50/101
            c5 = torch.randn(B, 2048, H, W, requires_grad=True)
            
            # Forward pass
            encoder_tokens = encoder(c5)
            
            # On s'attend à (B, H*W, 256)
            assert encoder_tokens.shape == (B, H*W, 256), (
                f"Shape tokens incorrect: {encoder_tokens.shape}, "
                f"attendu={(B, H*W, 256)}"
            )

            # Test de rétropropagation: on calcule une pseudo-loss et on backward
            loss = encoder_tokens.mean()
            loss.backward()  # aucune erreur ne doit être levée

    print("Test TransformerEncoder (layers + encoder) réussi ✅")

def test_backbone_afqs_encoder():
    """Test d'intégration pour backbone, encodeur et AFQS sans passer par QFreeDet."""
    # ------------------------------------------------------------------
    # 1. Initialisation des modules
    # ------------------------------------------------------------------
    backbone = ResNet50Backbone()
    encoder = TransformerEncoder()
    afqs = AFQS(max_pool_size=100)  # On limite à 100 pour le test

    # ------------------------------------------------------------------
    # 2. Données de test
    # ------------------------------------------------------------------
    batch_size = 2
    x = torch.randn(batch_size, 3, 640, 640)  # Images 640x640

    # ------------------------------------------------------------------
    # 3. Vérification de la sortie du backbone
    # ------------------------------------------------------------------
    features = backbone(x)
    assert len(features) == 4, (
        "Le backbone doit renvoyer 4 niveaux de features (c2, c3, c4, c5)."
    )
    c2, c3, c4, c5 = features

    # Pour une image 640x640, on s'attend à :
    # c2: 1/4  -> (B, 256, 160, 160)
    # c3: 1/8  -> (B, 512, 80, 80)
    # c4: 1/16 -> (B, 1024, 40, 40)
    # c5: 1/32 -> (B, 2048, 20, 20)
    assert c2.shape == (batch_size, 256, 160, 160), f"Shape de c2 incorrect: {c2.shape}"
    assert c3.shape == (batch_size, 512, 80, 80),   f"Shape de c3 incorrect: {c3.shape}"
    assert c4.shape == (batch_size, 1024, 40, 40),  f"Shape de c4 incorrect: {c4.shape}"
    assert c5.shape == (batch_size, 2048, 20, 20),  f"Shape de c5 incorrect: {c5.shape}"

    # ------------------------------------------------------------------
    # 4. Vérification de la sortie de l'encodeur
    # ------------------------------------------------------------------
    encoder_tokens = encoder(c5)  # [B, N, D] = [2, 400, 256] normalement
    assert encoder_tokens.shape == (batch_size, 400, 256), (
        f"Shape de la sortie de l'encodeur incorrecte: {encoder_tokens.shape}"
    )

    # ------------------------------------------------------------------
    # 5. Test du module AFQS en mode entraînement
    # ------------------------------------------------------------------
    sadq_train, mask_train = afqs(encoder_tokens)
    
    # Par défaut, en entraînement, la taille de SADQ devrait être [B, max_pool_size, D]
    # si des tokens sont sélectionnés (ou padding). Ici max_pool_size=100.
    assert sadq_train.shape == (batch_size, 100, 256), (
        f"Shape SADQ en mode entraînement incorrecte: {sadq_train.shape}"
    )
    assert mask_train.shape == (batch_size, 400), (
        f"Shape du masque de sélection incorrecte: {mask_train.shape}"
    )
    assert mask_train.dtype == torch.bool, (
        f"Le masque de sélection doit être booléen, obtenu: {mask_train.dtype}"
    )

    # ------------------------------------------------------------------
    # 6. Test du module AFQS en mode évaluation (inférence)
    # ------------------------------------------------------------------
    afqs.eval()  # Mode inférence
    with torch.no_grad():
        sadq_eval, mask_eval = afqs(encoder_tokens)

    # En inférence, AFQS renvoie une liste de tenseurs de taille variable
    # (un par image), ou un "collate" identique si on l’a implémenté différemment.
    # Dans le code actuel, c'est plutôt un list comprehension:
    #   SADQ = [encoder_tokens[b, selection_mask[b]] ...]
    # Donc on s'attend à une liste de length B.
    assert isinstance(sadq_eval, list), (
        "En mode inference, AFQS devrait renvoyer une liste de tenseurs."
    )
    assert len(sadq_eval) == batch_size, (
        f"En mode inférence, la liste renvoyée doit avoir {batch_size} éléments."
    )
    # Vérifions que la forme [N_i, D] est respectée pour chaque sous-tenseur
    for i, queries in enumerate(sadq_eval):
        assert queries.ndim == 2 and queries.shape[-1] == 256, (
            f"SADQ inférence (batch {i}) doit être [N_i, 256], obtenu {queries.shape}"
        )

    print("Test d'intégration (backbone / encodeur / AFQS) réussi ✅")

import torch

def test_qfreedet_integration():
    """Test d'intégration complet pour QFreeDet (backbone, encodeur, AFQS, BLP et DP) sur GPU si disponible."""
    torch.manual_seed(0)
    batch_size = 2
    img_size = 640  # images 640x640

    backbone = ResNet50Backbone().to(device)
    encoder = TransformerEncoder().to(device)
    afqs = AFQS(threshold=0.5, max_pool_size=100, num_classes=80, feature_dim=256).to(device)
    # Paramètres pour BLP et DP
    embed_dim = 256
    D = 256
    H, W = 20, 20
    T1 = 3  # nombre d'itérations pour BoxLocatingPart
    T2 = 3  # nombre d'itérations pour DeduplicationPart
    lambd = 1

    blp = BoxLocatingPart(embed_dim=embed_dim, H=H, W=W, D=D, T1=T1).to(device)
    dp = DeduplicationPart(embed_dim=embed_dim, H=H, W=W, D=D, T2=T2, lambd=lambd).to(device)

    model = QFreeDet(backbone=backbone, afqs=afqs, encoder=encoder, blp=blp, dp=dp).to(device)
    model.train()  # Test en mode entraînement

    x = torch.randn(batch_size, 3, img_size, img_size).to(device)
    B_blp, class_scores_blp, B_final, final_class_scores = model(x)

    # On attend pour chaque image 100 boîtes et 100 sets de scores de classes
    assert B_blp.shape == (batch_size, 100, 4), f"Shape de B_blp incorrecte: {B_blp.shape}"
    assert class_scores_blp.shape == (batch_size, 100, 80), f"Shape de class_scores_blp incorrecte: {class_scores_blp.shape}"
    assert B_final.shape == (batch_size, 100, 4), f"Shape de B_final incorrecte: {B_final.shape}"
    assert final_class_scores.shape == (batch_size, 100, 80), f"Shape de final_class_scores incorrecte: {final_class_scores.shape}"
    
    print("Test d'intégration QFreeDet réussi ✅")

def test_pocoo_loss():
    """Test de la fonction de perte PoCooLoss."""
    # ------------------------------------------------------------------
    # 1. Initialisation de la loss
    # ------------------------------------------------------------------
    loss_fn = PoCooLoss(alpha=0.5, iou_threshold=0.5)
    
    # ------------------------------------------------------------------
    # 2. Génération de données de test
    # ------------------------------------------------------------------
    B, M, J = 2, 100, 3  # Batch size, prédictions, ground truths
    H, W = 640, 640  # Taille de l'image
    
    B_final = torch.rand(B, M, 4) * torch.tensor([W, H, W, H])  # Boîtes prédites
    final_class_scores = torch.sigmoid(torch.randn(B, M, 80))  # Scores de classification
    gt_boxes = torch.rand(B, J, 4) * torch.tensor([W, H, W, H])  # Boîtes ground truth
    gt_labels = torch.randint(0, 2, (B, J, 80)).float()  # Labels ground truth
    
    # ------------------------------------------------------------------
    # 3. Calcul de la perte
    # ------------------------------------------------------------------
    loss_value = loss_fn(B_final, final_class_scores, gt_boxes, gt_labels, (H, W))
    
    # ------------------------------------------------------------------
    # 4. Vérifications de la sortie
    # ------------------------------------------------------------------
    assert isinstance(loss_value, torch.Tensor), f"La loss doit être un tenseur, obtenu: {type(loss_value)}"
    assert loss_value.dim() == 0, f"La loss doit être scalaire, obtenu: {loss_value.shape}"
    assert loss_value.item() >= 0, f"La loss doit être positive, obtenu: {loss_value.item()}"
    
    # ------------------------------------------------------------------
    # 5. Vérifications des masques et pondérations
    # ------------------------------------------------------------------
    iou_matrix = compute_iou(B_final, gt_boxes)  # [B, M, J]
    matched_iou, _ = iou_matrix.max(dim=-1)  # [B, M]
    
    pos_mask = (matched_iou > loss_fn.iou_threshold).float()
    neg_mask = (pos_mask == 0).float()
    
    assert torch.all((pos_mask + neg_mask) == 1), "Les masques positif et négatif doivent être complémentaires."
    
    print("Test de la PoCoo Loss réussi ✅")

def test_model_forward_and_loss():
    """
    Test complet avec calcul indépendant des losses pour BLP et DP.
    Vérifie que chaque partie du modèle reçoit bien ses gradients.
    """
    torch.manual_seed(42)
    batch_size = 1
    img_size = 640  # Taille des images (640x640)

    # Initialisation des composants du modèle sur le même device
    backbone = ResNet50Backbone().to(device)
    encoder = TransformerEncoder().to(device)
    afqs = AFQS(threshold=0.5, max_pool_size=100, num_classes=80, feature_dim=256).to(device)
    
    # Paramètres pour BoxLocatingPart (BLP) et DeduplicationPart (DP)
    embed_dim = 256
    D = 256
    H, W = 20, 20
    T1 = 3  # Nombre d'itérations pour BLP
    T2 = 3  # Nombre d'itérations pour DP
    lambd = 1
    
    blp = BoxLocatingPart(embed_dim=embed_dim, H=H, W=W, D=D, T1=T1).to(device)
    dp = DeduplicationPart(embed_dim=embed_dim, H=H, W=W, D=D, T2=T2, lambd=lambd).to(device)

    # Création du modèle QFreeDet
    model = QFreeDet(backbone=backbone, afqs=afqs, encoder=encoder, blp=blp, dp=dp).to(device)
    model.train()

    # Création d'un batch d'images synthétiques
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)

    # Génération de ground truths synthétiques
    num_gt = 5  # Nombre de ground truth par image
    gt_boxes = torch.rand(batch_size, num_gt, 4, device=device) * img_size
    num_classes = 80
    gt_labels = torch.zeros(batch_size, num_gt, num_classes, device=device)
    for i in range(batch_size):
        for j in range(num_gt):
            cls = torch.randint(0, num_classes, (1,), device=device).item()
            gt_labels[i, j, cls] = 1.0

    # Initialisation de la loss PoCooLoss
    loss_fn = PoCooLoss(alpha=0.5, iou_threshold=0.5).to(device)

    # === CALCUL DE LA LOSS POUR BLP ===
    optimizer_blp = torch.optim.Adam(model.blp.parameters(), lr=1e-4)
    B_blp, class_scores_blp = model(x, stage="blp")  # Exécute uniquement BLP
    loss_blp = loss_fn(B_blp, class_scores_blp, gt_boxes, gt_labels, (img_size, img_size))

    optimizer_blp.zero_grad()
    loss_blp.backward()
    optimizer_blp.step()

    # === CALCUL DE LA LOSS POUR DP ===
    optimizer_dp = torch.optim.Adam(model.dp.parameters(), lr=1e-4)
    B_final, final_class_scores = model(x, stage="dp")  # Exécute uniquement DP
    loss_dp = loss_fn(B_final, final_class_scores, gt_boxes, gt_labels, (img_size, img_size))

    optimizer_dp.zero_grad()
    loss_dp.backward()
    optimizer_dp.step()

    # === Vérification que chaque module reçoit bien ses gradients ===
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            raise RuntimeError(f"Gradient manquant pour le paramètre : {name}")
    
    print(f"Test modèle forward et calcul de loss réussi ✅")
    print(f"Loss BLP: {loss_blp.item():.4f} | Loss DP: {loss_dp.item():.4f}")

if __name__ == "__main__":
    
    test_afqs()
    test_dataset_and_backbone()
    
    test_positional_encoding()
    test_multihead_attention()
    test_deformable_attention()
    test_transformer_encoder()
    test_backbone_afqs_encoder()
    test_qfreedet_integration()
    test_pocoo_loss()
    test_model_forward_and_loss() 
    test_visualization()