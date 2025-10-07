import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

class NeuralFeatureExtractor:
    """
    Classe pour l'extraction de caractéristiques et la classification avec réseaux de neurones
    """
    
    def __init__(self, max_vocab_size=1000, max_seq_length=100):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.le_browser = LabelEncoder()
        self.le_target = LabelEncoder()
        self.scaler = StandardScaler()
        self.vocab = None
        self.action_to_idx = None
        self.mlp = None
        
    def prepare_data(self, df):
        """
        Prépare les données pour l'entraînement des réseaux de neurones
        """
        print("Préparation des données...")
        
        # 1. Encoder les navigateurs
        browser_encoded = self.le_browser.fit_transform(df['navigateur'])
        
        # 2. Extraire les actions (colonnes 2 et plus)
        actions_df = df.iloc[:, 2:].copy()
        
        # 3. Remplacer les NaN par une valeur spéciale
        actions_df = actions_df.fillna('PAD')
        
        print(f"Forme des données d'actions: {actions_df.shape}")
        print(f"Navigateurs uniques: {len(self.le_browser.classes_)}")
        
        return actions_df, browser_encoded
    
    def extract_basic_features(self, actions_df, browser_encoded):
        """
        Extrait des caractéristiques basiques pour les réseaux de neurones
        """
        features_list = []
        
        print("Extraction des caractéristiques basiques...")
        
        for idx, row in actions_df.iterrows():
            # Filtrer les actions non-nulles
            valid_actions = [action for action in row if action != 'PAD']
            
            # Caractéristiques de base
            session_length = len(valid_actions)
            
            # Compter les actions uniques
            action_counts = Counter(valid_actions)
            unique_actions = len(action_counts)
            
            # Compter les marqueurs temporels
            temporal_markers = sum(1 for action in valid_actions if str(action).startswith('t'))
            
            # Actions les plus fréquentes (top 3)
            most_common = action_counts.most_common(3)
            top_action_freq = most_common[0][1] if most_common else 0
            action_diversity = unique_actions / max(session_length, 1)
            
            # Caractéristiques de séquence
            first_action = str(valid_actions[0]) if valid_actions else 'EMPTY'
            last_action = str(valid_actions[-1]) if valid_actions else 'EMPTY'
            
            # Patterns d'activité
            time_ratio = temporal_markers / max(session_length, 1)
            
            features = {
                'session_length': session_length,
                'unique_actions': unique_actions,
                'temporal_markers': temporal_markers,
                'top_action_frequency': top_action_freq,
                'action_diversity': action_diversity,
                'time_ratio': time_ratio,
                'browser': browser_encoded[idx],
                'first_action_hash': hash(first_action) % 1000,
                'last_action_hash': hash(last_action) % 1000
            }
            
            features_list.append(features)
            
            if (idx + 1) % 500 == 0:
                print(f"Traité {idx + 1} sessions...")
        
        return pd.DataFrame(features_list)
    
    def create_action_vocabulary(self, actions_df):
        """
        Crée un vocabulaire des actions les plus fréquentes
        """
        print("Création du vocabulaire d'actions...")
        
        all_actions = []
        for _, row in actions_df.iterrows():
            valid_actions = [action for action in row if action != 'PAD']
            all_actions.extend(valid_actions)
        
        # Compter les actions
        action_counts = Counter(all_actions)
        
        # Garder les actions les plus fréquentes
        self.vocab = ['PAD', 'UNK'] + [action for action, _ in action_counts.most_common(self.max_vocab_size-2)]
        self.action_to_idx = {action: idx for idx, action in enumerate(self.vocab)}
        
        print(f"Vocabulaire créé: {len(self.vocab)} actions uniques")
        return self.vocab, self.action_to_idx
    
    def extract_sequence_features(self, actions_df):
        """
        Extrait des caractéristiques de séquence pour les réseaux de neurones
        """
        print("Extraction des caractéristiques de séquence...")
        
        sequences = []
        
        for idx, row in actions_df.iterrows():
            valid_actions = [action for action in row if action != 'PAD']
            
            # Convertir en indices
            sequence = []
            for action in valid_actions[:self.max_seq_length]:
                if str(action) in self.action_to_idx:
                    sequence.append(self.action_to_idx[str(action)])
                else:
                    sequence.append(self.action_to_idx['UNK'])
            
            # Padding pour avoir des séquences de même longueur
            while len(sequence) < self.max_seq_length:
                sequence.append(self.action_to_idx['PAD'])
            
            sequences.append(sequence)
            
            if (idx + 1) % 500 == 0:
                print(f"Traité {idx + 1} séquences...")
        
        return np.array(sequences)
    
    def sequence_to_stats(self, sequences):
        """Convertit les séquences en statistiques numériques"""
        stats = []
        for seq in sequences:
            seq_array = np.array(seq)
            # Enlever le padding
            non_pad = seq_array[seq_array != self.action_to_idx['PAD']]
            if len(non_pad) > 0:
                stats.append([
                    np.mean(non_pad),
                    np.std(non_pad),
                    np.min(non_pad),
                    np.max(non_pad),
                    len(non_pad),
                    len(np.unique(non_pad))
                ])
            else:
                stats.append([0, 0, 0, 0, 0, 0])
        return np.array(stats)
    
    def fit_transform(self, df):
        """
        Extrait toutes les caractéristiques et prépare les données pour l'entraînement
        """
        print("=== EXTRACTION COMPLÈTE DES CARACTÉRISTIQUES ===")
        
        # Préparer les données
        actions_df, browser_encoded = self.prepare_data(df)
        
        # Extraire les caractéristiques basiques
        basic_features = self.extract_basic_features(actions_df, browser_encoded)
        
        # Créer le vocabulaire et extraire les séquences
        self.create_action_vocabulary(actions_df)
        sequence_features = self.extract_sequence_features(actions_df)
        
        # Convertir les séquences en statistiques
        sequence_stats = self.sequence_to_stats(sequence_features)
        
        # Normaliser les caractéristiques basiques
        basic_features_scaled = self.scaler.fit_transform(basic_features)
        
        # Combiner toutes les caractéristiques
        X_combined = np.hstack([basic_features_scaled, sequence_stats])
        
        # Encoder la variable cible
        y_encoded = self.le_target.fit_transform(df['util'])
        
        print(f"Caractéristiques finales: {X_combined.shape}")
        print(f"Nombre de classes: {len(self.le_target.classes_)}")
        
        return X_combined, y_encoded, basic_features
    
    def transform(self, df):
        """
        Transforme de nouvelles données en utilisant les paramètres déjà appris
        """
        print("=== TRANSFORMATION DES DONNÉES DE TEST ===")
        
        # Préparer les données
        actions_df = df.iloc[:, 2:].copy().fillna('PAD')
        browser_encoded = self.le_browser.transform(df['navigateur'])
        
        # Extraire les caractéristiques basiques
        basic_features = self.extract_basic_features(actions_df, browser_encoded)
        
        # Extraire les séquences avec le vocabulaire existant
        sequence_features = self.extract_sequence_features(actions_df)
        sequence_stats = self.sequence_to_stats(sequence_features)
        
        # Normaliser avec le scaler déjà ajusté
        basic_features_scaled = self.scaler.transform(basic_features)
        
        # Combiner les caractéristiques
        X_combined = np.hstack([basic_features_scaled, sequence_stats])
        
        return X_combined
    
    def train_neural_network(self, X_train, y_train, hidden_layers=(256, 128, 64)):
        """
        Entraîne le réseau de neurones
        """
        print("=== ENTRAÎNEMENT DU RÉSEAU DE NEURONES ===")
        
        self.mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=True
        )
        
        print("Entraînement en cours...")
        self.mlp.fit(X_train, y_train)
        
        print(f"Nombre d'itérations: {self.mlp.n_iter_}")
        return self.mlp
    
    def evaluate_model(self, X_test, y_test):
        """
        Évalue le modèle sur les données de test
        """
        print("=== ÉVALUATION DU MODÈLE ===")
        
        from sklearn.metrics import f1_score, classification_report
        
        y_pred = self.mlp.predict(X_test)
        y_pred_proba = self.mlp.predict_proba(X_test)
        
        # Précision globale
        accuracy = self.mlp.score(X_test, y_test)
        
        # F1-score (macro et weighted pour classes déséquilibrées)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        
        print(f"Précision globale: {accuracy:.4f}")
        print(f"F1-score macro: {f1_macro:.4f}")
        print(f"F1-score weighted: {f1_weighted:.4f}")
        print(f"F1-score micro: {f1_micro:.4f}")
        
        # Stocker le F1-score pour utilisation ultérieure
        self.f1 = f1_weighted  # ou f1_macro selon votre préférence
        
        # Top-k accuracy
        def top_k_accuracy(y_true, y_pred_proba, k=5):
            top_k_pred = np.argsort(y_pred_proba, axis=1)[:, -k:]
            correct = 0
            for i, true_label in enumerate(y_true):
                if true_label in top_k_pred[i]:
                    correct += 1
            return correct / len(y_true)
        
        for k in [1, 3, 5, 10]:
            acc_k = top_k_accuracy(y_test, y_pred_proba, k)
            print(f"Top-{k} accuracy: {acc_k:.4f}")
        
        # Afficher un rapport de classification détaillé
        print("\n=== RAPPORT DE CLASSIFICATION ===")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        
        return y_pred, y_pred_proba, f1_weighted

    def plot_results(self, y_test, y_pred, y_pred_proba, basic_features):
        """
        Visualise les résultats du modèle
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Courbe de perte
        axes[0,0].plot(self.mlp.loss_curve_)
        axes[0,0].set_title('Courbe de perte pendant l\'entraînement')
        axes[0,0].set_xlabel('Itération')
        axes[0,0].set_ylabel('Perte')
        axes[0,0].grid(True)
        
        # 2. Distribution des probabilités de prédiction
        max_probs = np.max(y_pred_proba, axis=1)
        axes[0,1].hist(max_probs, bins=50, alpha=0.7, edgecolor='black')
        axes[0,1].set_title('Distribution des probabilités maximales')
        axes[0,1].set_xlabel('Probabilité maximale')
        axes[0,1].set_ylabel('Fréquence')
        
        # 3. Matrice de confusion (échantillon)
        unique_classes = np.unique(np.concatenate([y_test, y_pred]))[:20]
        y_test_sample = y_test[np.isin(y_test, unique_classes)]
        y_pred_sample = y_pred[np.isin(y_test, unique_classes)]
        
        if len(y_test_sample) > 0:
            cm = confusion_matrix(y_test_sample, y_pred_sample)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[1,0], cmap='Blues')
            axes[1,0].set_title('Matrice de confusion (échantillon)')
            axes[1,0].set_xlabel('Prédictions')
            axes[1,0].set_ylabel('Vraies valeurs')
        
        # 4. Importance des caractéristiques
        feature_names = (list(basic_features.columns) + 
                        ['seq_mean', 'seq_std', 'seq_min', 'seq_max', 'seq_len', 'seq_unique'])
        
        if hasattr(self.mlp, 'coefs_'):
            feature_importance = np.var(self.mlp.coefs_[0], axis=1)
            top_features = np.argsort(feature_importance)[-10:]
            
            axes[1,1].barh(range(len(top_features)), feature_importance[top_features])
            axes[1,1].set_yticks(range(len(top_features)))
            axes[1,1].set_yticklabels([feature_names[i] for i in top_features])
            axes[1,1].set_title('Importance approximative des caractéristiques')
            axes[1,1].set_xlabel('Variance des poids')
        
        plt.tight_layout()
        plt.show()