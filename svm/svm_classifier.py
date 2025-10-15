from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class SVMUserClassifier:
    """
    Classificateur SVM optimisé pour la prédiction d'utilisateurs
    """
    
    def __init__(self, kernel='rbf', use_grid_search=True):
        self.kernel = kernel
        self.use_grid_search = use_grid_search
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.svm = None
        self.best_params = None
        
    def train(self, X, y, verbose=True):
        """
        Entraîne le modèle SVM avec ou sans grid search
        """
        print("=== ENTRAÎNEMENT SVM ===")
        print(f"Kernel: {self.kernel}")
        print(f"Échantillons: {X.shape[0]}")
        print(f"Features: {X.shape[1]}")
        print(f"Classes: {len(np.unique(y))}")
        
        # Normalisation des features (CRUCIAL pour SVM)
        X_scaled = self.scaler.fit_transform(X)
        
        # Encoder les labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        if self.use_grid_search:
            print("\n>>> Grid Search en cours...")
            self._grid_search_train(X_scaled, y_encoded, verbose)
        else:
            print("\n>>> Entraînement avec paramètres par défaut...")
            self._default_train(X_scaled, y_encoded)
        
        # Évaluation par cross-validation
        print("\n>>> Cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.svm, X_scaled, y_encoded, 
                                     cv=cv, scoring='f1_weighted', n_jobs=-1)
        
        print(f"\nF1 Score (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Entraînement final sur toutes les données
        print("\n>>> Entraînement final...")
        self.svm.fit(X_scaled, y_encoded)
        
        return self
    
    def _grid_search_train(self, X_scaled, y_encoded, verbose):
        """
        Recherche des meilleurs hyperparamètres
        """
        # Paramètres à tester selon le kernel
        if self.kernel == 'rbf':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'class_weight': ['balanced', None]
            }
        elif self.kernel == 'poly':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'degree': [2, 3, 4],
                'gamma': ['scale', 'auto'],
                'class_weight': ['balanced', None]
            }
        elif self.kernel == 'linear':
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'class_weight': ['balanced', None]
            }
        else:
            param_grid = {
                'C': [0.1, 1, 10],
                'class_weight': ['balanced']
            }
        
        # Grid Search avec cross-validation
        grid_search = GridSearchCV(
            SVC(kernel=self.kernel, random_state=42),
            param_grid,
            cv=3,  # 3 folds pour gagner du temps
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=2 if verbose else 0
        )
        
        grid_search.fit(X_scaled, y_encoded)
        
        self.svm = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"\nMeilleurs paramètres: {self.best_params}")
        print(f"Meilleur score F1: {grid_search.best_score_:.4f}")
    
    def _default_train(self, X_scaled, y_encoded):
        """
        Entraînement avec paramètres par défaut optimisés
        """
        if self.kernel == 'rbf':
            self.svm = SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                class_weight='balanced',
                random_state=42
            )
        elif self.kernel == 'poly':
            self.svm = SVC(
                kernel='poly',
                degree=3,
                C=10,
                gamma='scale',
                class_weight='balanced',
                random_state=42
            )
        elif self.kernel == 'linear':
            self.svm = SVC(
                kernel='linear',
                C=1,
                class_weight='balanced',
                random_state=42
            )
        else:
            self.svm = SVC(
                kernel=self.kernel,
                class_weight='balanced',
                random_state=42
            )
        
        self.svm.fit(X_scaled, y_encoded)
    
    def predict(self, X):
        """
        Prédit les classes
        """
        X_scaled = self.scaler.transform(X)
        y_pred_encoded = self.svm.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def evaluate(self, X_test, y_test):
        """
        Évalue le modèle sur un ensemble de test
        """
        print("\n=== ÉVALUATION DU MODÈLE ===")
        
        y_pred = self.predict(X_test)
        
        # F1 Score
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        print(f"\nF1 Score (weighted): {f1_weighted:.4f}")
        print(f"F1 Score (macro): {f1_macro:.4f}")
        
        # Rapport de classification
        print("\n" + classification_report(y_test, y_pred))
        
        return f1_weighted, y_pred
    
    def plot_confusion_matrix(self, y_test, y_pred, top_n=20):
        """
        Affiche la matrice de confusion pour les N classes les plus fréquentes
        """
        # Sélectionner les top N classes
        top_classes = pd.Series(y_test).value_counts().head(top_n).index
        
        # Filtrer
        mask = pd.Series(y_test).isin(top_classes)
        y_test_filtered = np.array(y_test)[mask]
        y_pred_filtered = np.array(y_pred)[mask]
        
        # Matrice de confusion
        cm = confusion_matrix(y_test_filtered, y_pred_filtered, 
                             labels=top_classes)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=top_classes, yticklabels=top_classes)
        plt.title(f'Matrice de Confusion - Top {top_n} Utilisateurs')
        plt.ylabel('Vraie Classe')
        plt.xlabel('Classe Prédite')
        plt.tight_layout()
        plt.show()