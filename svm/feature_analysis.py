import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

class FeatureAnalyzer:
    """
    Analyse la corrélation et l'importance des features
    """
    
    def __init__(self, threshold_corr=0.85, threshold_importance=0.001):
        self.threshold_corr = threshold_corr
        self.threshold_importance = threshold_importance
        self.features_to_keep = None
        self.feature_importance = None
        
    def analyze_correlations(self, X, plot=True):
        """
        Trouve les features fortement corrélées
        """
        print("\n=== ANALYSE DES CORRÉLATIONS ===")
        
        # Matrice de corrélation
        corr_matrix = X.corr().abs()
        
        # Triangle supérieur
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Trouver les features à supprimer
        to_drop = []
        for column in upper_triangle.columns:
            correlated_features = upper_triangle[column][upper_triangle[column] > self.threshold_corr]
            if len(correlated_features) > 0:
                to_drop.append(column)
                print(f"\n❌ {column}")
                for feat, corr_val in correlated_features.items():
                    print(f"   Corrélé avec {feat}: {corr_val:.3f}")
        
        print(f"\n✓ Features à supprimer (corrélation > {self.threshold_corr}): {len(to_drop)}")
        
        if plot and len(corr_matrix) < 100:
            # Heatmap des corrélations (seulement pour les features non-TF-IDF)
            non_tfidf = [col for col in X.columns if not col.startswith('tfidf_')]
            if len(non_tfidf) < 80:
                plt.figure(figsize=(20, 16))
                sns.heatmap(X[non_tfidf].corr(), annot=False, cmap='coolwarm', center=0)
                plt.title('Matrice de Corrélation des Features')
                plt.tight_layout()
                plt.show()
        
        return to_drop
    
    def analyze_importance(self, X, y, plot=True):
        """
        Calcule l'importance des features avec Mutual Information
        """
        print("\n=== ANALYSE DE L'IMPORTANCE ===")
        
        # Mutual Information (mesure la dépendance entre features et target)
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Créer DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        
        # Features peu importantes
        low_importance = importance_df[importance_df['importance'] < self.threshold_importance]
        
        print(f"\n✓ Top 20 features les plus importantes:")
        print(importance_df.head(20).to_string(index=False))
        
        print(f"\n❌ Features peu importantes (MI < {self.threshold_importance}): {len(low_importance)}")
        print(low_importance.head(10).to_string(index=False))
        
        if plot:
            # Plot top 30
            plt.figure(figsize=(12, 8))
            top_30 = importance_df.head(30)
            plt.barh(range(len(top_30)), top_30['importance'])
            plt.yticks(range(len(top_30)), top_30['feature'])
            plt.xlabel('Mutual Information Score')
            plt.title('Top 30 Features par Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
        
        return low_importance['feature'].tolist()
    
    def select_features(self, X, y, plot=True):
        """
        Sélectionne les meilleures features
        """
        print("\n" + "="*60)
        print("SÉLECTION AUTOMATIQUE DES FEATURES")
        print("="*60)
        
        # 1. Supprimer features corrélées
        correlated = self.analyze_correlations(X, plot=plot)
        
        # 2. Supprimer features peu importantes
        low_importance = self.analyze_importance(X, y, plot=plot)
        
        # Combiner
        to_remove = set(correlated + low_importance)
        self.features_to_keep = [col for col in X.columns if col not in to_remove]
        
        print("\n" + "="*60)
        print("RÉSUMÉ")
        print("="*60)
        print(f"Features originales: {len(X.columns)}")
        print(f"Features corrélées supprimées: {len(correlated)}")
        print(f"Features peu importantes supprimées: {len(low_importance)}")
        print(f"Features conservées: {len(self.features_to_keep)}")
        print(f"Réduction: {(1 - len(self.features_to_keep)/len(X.columns))*100:.1f}%")
        
        return self.features_to_keep