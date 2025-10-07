# Ajouter cette cellule après la section "Construction de caractéristiques"
from neural_feature_extractor import NeuralFeatureExtractor

# =============================================================================
# CONSTRUCTION DE CARACTÉRISTIQUES AVEC RÉSEAUX DE NEURONES
# =============================================================================

# Créer l'extracteur de caractéristiques
feature_extractor = NeuralFeatureExtractor(max_vocab_size=1000, max_seq_length=100)

# Extraire toutes les caractéristiques
X_combined, y_encoded, basic_features = feature_extractor.fit_transform(features_train)

# Division train/test stratifiée
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

print(f"Données d'entraînement: {X_train.shape}")
print(f"Données de test: {X_test.shape}")

# Entraîner le réseau de neurones
model = feature_extractor.train_neural_network(X_train, y_train)

# Évaluer le modèle
y_pred, y_pred_proba, accuracy = feature_extractor.evaluate_model(X_test, y_test)

# Visualiser les résultats
feature_extractor.plot_results(y_test, y_pred, y_pred_proba, basic_features)

# Afficher le résumé
print("\n=== RÉSUMÉ DE LA CONSTRUCTION DE CARACTÉRISTIQUES ===")
print(f"✓ Caractéristiques basiques extraites: {basic_features.shape[1]}")
print(f"✓ Caractéristiques de séquence: 6")
print(f"✓ Total des caractéristiques: {X_combined.shape[1]}")
print(f"✓ Modèle entraîné avec {model.n_iter_} itérations")
print(f"✓ Précision finale: {accuracy:.4f}")


