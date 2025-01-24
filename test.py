import torch

from model import AFQS

def test_afqs():
    # Paramètres communs
    feature_dim = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
if __name__ == "__main__":
    test_afqs()