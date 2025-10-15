import pandas as pd
import numpy as np
from feature_extractor import FeatureExtractor
from feature_analysis import FeatureAnalyzer
from svm_classifier import SVMUserClassifier
from sklearn.model_selection import train_test_split

# 1. CHARGEMENT
print("=== CHARGEMENT DES DONNÉES ===")

def read_ds(ds_name: str):
    df = pd.read_csv(ds_name, header=None, engine='python', on_bad_lines='skip')
    if 'train' in ds_name:
        df = df.rename(columns={0: 'util', 1: 'navigateur'})
    else:
        df = df.rename(columns={0: 'navigateur'})
    return df

train_df = read_ds("data/train.csv")
test_df = read_ds("data/test.csv")
print(f"Train: {train_df.shape}, Utilisateurs: {train_df['util'].nunique()}")
print(f"Test: {test_df.shape}")

# print(f"Train: {train_df.shape}, Utilisateurs: {train_df['util'].nunique()}")

# # 2. EXTRACTION FEATURES OPTIMISÉES
# print("\n=== EXTRACTION FEATURES OPTIMISÉES ===")
# extractor = FeatureExtractor()

# X_train_full = extractor.extract_features(train_df, is_train=True)
# y_train_full = train_df['util']

# print(f"Features extraites: {X_train_full.shape[1]}")

# # 3. ANALYSE DES FEATURES
# print("\n=== ANALYSE ET SÉLECTION DES FEATURES ===")
# analyzer = FeatureAnalyzer(threshold_corr=0.85, threshold_importance=0.001)

# # Sélectionner les meilleures features
# best_features = analyzer.select_features(X_train_full, y_train_full, plot=True)

# # Filtrer
# X_train_full_selected = X_train_full[best_features]

# print(f"\n✓ Features après sélection: {X_train_full_selected.shape[1]}")

# # 4. SPLIT
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train_full_selected, y_train_full, 
#     test_size=0.2, 
#     stratify=y_train_full,
#     random_state=42
# )

# # 5. ENTRAÎNEMENT
# print("\n" + "="*60)
# print("ENTRAÎNEMENT SVM")
# print("="*60)

# svm = SVMUserClassifier(kernel='rbf', use_grid_search=True)
# svm.train(X_train, y_train, verbose=True)

# # 6. ÉVALUATION
# f1_val, y_pred_val = svm.evaluate(X_val, y_val)
# print(f"\n✓ F1 Score (Validation): {f1_val:.4f}")

# # 7. PRÉDICTION TEST
# print("\n=== PRÉDICTION SUR TEST ===")
# X_test = extractor.extract_features(test_df, is_train=False)
# X_test_selected = X_test[best_features]

# predictions = svm.predict(X_test_selected)

# # 8. SOUMISSION
# submission = pd.DataFrame({
#     'RowId': range(len(predictions)),
#     'util': predictions
# })
# submission.to_csv('submissioncsv', index=False)