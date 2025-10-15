import pandas as pd
import numpy as np
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    """
    Extracteur de features optimisé - suppression des redondances
    Focus sur les RATIOS et FRÉQUENCES plutôt que les QUANTITÉS absolues
    """
    
    def __init__(self):
        self.pattern_ecran = re.compile(r"\((.*?)\)")
        self.pattern_conf_ecran = re.compile(r"<(.*?)>")
        self.pattern_chaine = re.compile(r"\$(.*?)\$")
        
        # TF-IDF réduit (moins de features)
        self.tfidf_actions = TfidfVectorizer(
            max_features=50,  # Réduit de 100 à 50
            ngram_range=(1, 2),
            analyzer='word',
            token_pattern=r'[^,]+'
        )
        
        self.tfidf_ecrans = TfidfVectorizer(
            max_features=30,  # Réduit de 50 à 30
            analyzer='word',
            token_pattern=r'\S+'
        )
        
        self.is_fitted = False
        
    def extract_features(self, df, is_train=True):
        """
        Extrait des features NON-CORRÉLÉES et PERTINENTES
        """
        features_list = []
        action_sequences = []
        ecran_sequences = []
        
        for idx, row in df.iterrows():
            if idx % 500 == 0:
                print(f"Processing {idx}/{len(df)}")
            
            # Récupérer données
            if is_train:
                navigateur = row['navigateur']
                actions = [str(a) for a in row.iloc[2:] if pd.notna(a)]
            else:
                navigateur = row.iloc[0]
                actions = [str(a) for a in row.iloc[1:] if pd.notna(a)]
            
            features = {}
            
            # Séparer temporel et non-temporel
            non_temporal = [a for a in actions if not a.startswith('t')]
            temporal_markers = [a for a in actions if a.startswith('t')]
            
            # Pour TF-IDF
            action_sequences.append(' '.join(non_temporal))
            
            # === 1. NAVIGATEUR (4 features) ===
            features['browser_firefox'] = 1 if navigateur == 'Firefox' else 0
            features['browser_chrome'] = 1 if navigateur == 'Google Chrome' else 0
            features['browser_edge'] = 1 if navigateur == 'Microsoft Edge' else 0
            features['browser_opera'] = 1 if navigateur == 'Opera' else 0
            
            # === 2. TEMPOREL - SIMPLIFIÉ (3 features au lieu de 12) ===
            time_values = []
            for marker in temporal_markers:
                try:
                    time_values.append(int(marker[1:]))
                except:
                    pass
            
            # ❌ SUPPRIMÉ: session_duration (trop variable selon le contexte)
            # ✅ GARDÉ: Seulement les features de RYTHME
            if len(time_values) > 1:
                time_diffs = np.diff(time_values)
                features['avg_action_speed'] = np.mean(time_diffs)  # Vitesse moyenne
                features['time_regularity'] = 1 / (np.std(time_diffs) + 1)  # Régularité (inverse de std)
            else:
                features['avg_action_speed'] = 0
                features['time_regularity'] = 0
            
            features['session_intensity'] = len(non_temporal) / (len(time_values) + 1)  # Actions par marqueur temporel
            
            # === 3. ACTIONS - FRÉQUENCES UNIQUEMENT (pas de quantités) ===
            total = len(non_temporal) if non_temporal else 1
            
            # ❌ SUPPRIMÉ: total_actions, nb_bouton, nb_saisie_champ, etc.
            # ✅ GARDÉ: Seulement les RATIOS (fréquences)
            features['freq_bouton'] = sum(1 for a in non_temporal if 'bouton' in a.lower()) / total
            features['freq_saisie'] = sum(1 for a in non_temporal if 'saisie' in a.lower()) / total
            features['freq_double_clic'] = sum(1 for a in non_temporal if 'double-clic' in a.lower()) / total
            features['freq_creation'] = sum(1 for a in non_temporal if 'création' in a.lower()) / total
            features['freq_modification'] = sum(1 for a in non_temporal if a.endswith('1')) / total
            features['freq_chainage'] = sum(1 for a in non_temporal if 'chainage' in a.lower()) / total
            features['freq_validation'] = sum(1 for a in non_temporal if 'validation' in a.lower()) / total
            
            # === 4. ÉCRANS - DIVERSITÉ ET PATTERNS ===
            ecrans = []
            for action in non_temporal:
                matches = self.pattern_ecran.findall(action)
                ecrans.extend(matches)
            
            ecran_sequences.append(' '.join(ecrans))
            
            # ❌ SUPPRIMÉ: nb_ecrans_visited (absolu)
            # ✅ GARDÉ: Diversité et concentration
            features['ecran_diversity'] = len(set(ecrans)) / len(ecrans) if ecrans else 0
            
            # Concentration sur les top écrans (mesure de spécialisation)
            if ecrans:
                ecran_counter = Counter(ecrans)
                top_1_ratio = ecran_counter.most_common(1)[0][1] / len(ecrans)
                top_3_total = sum([count for _, count in ecran_counter.most_common(3)])
                top_3_ratio = top_3_total / len(ecrans)
                
                features['ecran_concentration_top1'] = top_1_ratio
                features['ecran_concentration_top3'] = top_3_ratio
            else:
                features['ecran_concentration_top1'] = 0
                features['ecran_concentration_top3'] = 0
            
            # === 5. CONFIGURATIONS - SIMPLIFIÉ ===
            confs = []
            for action in non_temporal:
                matches = self.pattern_conf_ecran.findall(action)
                confs.extend(matches)
            
            # ❌ SUPPRIMÉ: nb_conf_changes (absolu)
            # ✅ GARDÉ: Diversité uniquement
            features['conf_diversity'] = len(set(confs)) / len(confs) if confs else 0
            
            # === 6. CHAÎNES - SIMPLIFIÉ ===
            chaines = []
            for action in non_temporal:
                matches = self.pattern_chaine.findall(action)
                chaines.extend(matches)
            
            features['chaine_diversity'] = len(set(chaines)) / len(chaines) if chaines else 0
            
            # === 7. COMPLEXITÉ COMPORTEMENTALE ===
            action_types = [a.split('(')[0].split('<')[0].split('$')[0].strip() for a in non_temporal]
            
            # Diversité des actions
            features['action_diversity'] = len(set(action_types)) / len(action_types) if action_types else 0
            
            # Entropie (mesure de prévisibilité)
            if action_types:
                action_probs = np.array(list(Counter(action_types).values())) / len(action_types)
                features['action_entropy'] = -np.sum(action_probs * np.log2(action_probs + 1e-10))
            else:
                features['action_entropy'] = 0
            
            # Répétitivité (inverse de la diversité)
            if len(action_types) > 1:
                action_counter = Counter(action_types)
                features['action_repetitiveness'] = max(action_counter.values()) / len(action_types)
            else:
                features['action_repetitiveness'] = 0
            
            # === 8. PATTERNS SÉQUENTIELS ===
            if len(action_types) > 1:
                # Transitions uniques (mesure de variété de workflow)
                transitions = [f"{action_types[i]}->{action_types[i+1]}" 
                              for i in range(len(action_types)-1)]
                features['transition_diversity'] = len(set(transitions)) / len(transitions)
            else:
                features['transition_diversity'] = 0
            
            # === 9. RYTHME - DISTRIBUTION TEMPORELLE ===
            if time_values and len(time_values) > 2:
                total_duration = max(time_values)
                if total_duration > 0:
                    third = total_duration / 3
                    actions_first = sum(1 for t in time_values if t < third)
                    actions_last = sum(1 for t in time_values if t >= 2*third)
                    
                    total_timed = len(time_values)
                    # Début vs Fin (pattern de démarrage)
                    features['start_intensity_ratio'] = actions_first / total_timed
                    features['end_intensity_ratio'] = actions_last / total_timed
                else:
                    features['start_intensity_ratio'] = 0
                    features['end_intensity_ratio'] = 0
            else:
                features['start_intensity_ratio'] = 0
                features['end_intensity_ratio'] = 0
            
            features_list.append(features)
        
        # Créer DataFrame de base
        df_features = pd.DataFrame(features_list)
        
        # === AJOUTER TF-IDF (réduit) ===
        if is_train:
            tfidf_actions_features = self.tfidf_actions.fit_transform(action_sequences).toarray()
            tfidf_ecrans_features = self.tfidf_ecrans.fit_transform(ecran_sequences).toarray()
            self.is_fitted = True
        else:
            tfidf_actions_features = self.tfidf_actions.transform(action_sequences).toarray()
            tfidf_ecrans_features = self.tfidf_ecrans.transform(ecran_sequences).toarray()
        
        # Créer colonnes TF-IDF
        tfidf_actions_df = pd.DataFrame(
            tfidf_actions_features,
            columns=[f'tfidf_action_{i}' for i in range(tfidf_actions_features.shape[1])]
        )
        
        tfidf_ecrans_df = pd.DataFrame(
            tfidf_ecrans_features,
            columns=[f'tfidf_ecran_{i}' for i in range(tfidf_ecrans_features.shape[1])]
        )
        
        # Combiner
        df_final = pd.concat([df_features, tfidf_actions_df, tfidf_ecrans_df], axis=1)
        
        print(f"\n✓ Features totales: {df_final.shape[1]} (optimisé)")
        
        return df_final