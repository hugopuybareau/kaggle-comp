import re
import pandas as pd
from collections import Counter

class FeatureEngineer:
    """L'objectif de cette classe est d'extraire des features pertinentes à partir des données brutes."""
    
    def __init__(self):
        # Regex patterns for extracting structured information
        self.pattern_ecran = re.compile(r"\((.*?)\)")
        self.pattern_conf_ecran = re.compile(r"<(.*?)>")
        self.pattern_chaine = re.compile(r"\$(.*?)\$")
        
    def filter_action(self, value):
        """Extract base action name without parameters"""
        if pd.isna(value):
            return None
        value = str(value)
        for delim in ["(", "<", "$", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]:
            if delim in value:
                low_ind = value.index(delim)
                return value[:low_ind]
        return value
    
    def extract_session_actions(self, row, start_col=2):
        """Extract all actions from a session row"""
        actions = []
        for val in row.iloc[start_col:]:
            if pd.notna(val) and not str(val).startswith('t'):
                actions.append(str(val))
        return actions
    
    def extract_temporal_info(self, row, start_col=2):
        """Extract temporal markers (tXX) to measure time"""
        time_markers = []
        for val in row.iloc[start_col:]:
            if pd.notna(val) and str(val).startswith('t'):
                time_markers.append(str(val))
        return time_markers
    
    def extract_features(self, df, is_train=True):
        """
        Extract comprehensive features from raw data
        
        Returns DataFrame with features:
        - Basic session metrics (length, duration)
        - Action frequencies
        - Screen/configuration usage
        - Temporal patterns
        - Browser encoding
        """
        
        print("Extracting features...")
        
        start_col = 2 if is_train else 1
        features = {}
        
        # ---- BASIC SESSION FEATURES ----
        print("  - Session length and duration features...")
        
        # Number of actions (excluding time markers)
        features['num_actions'] = df.iloc[:, start_col:].apply(
            lambda row: sum(pd.notna(val) and not str(val).startswith('t') 
                           for val in row), axis=1
        )
        
        # Number of time markers (5-second intervals)
        features['num_time_markers'] = df.iloc[:, start_col:].apply(
            lambda row: sum(pd.notna(val) and str(val).startswith('t') 
                           for val in row), axis=1
        )
        
        # Estimated session duration (in 5-second intervals)
        features['session_duration'] = features['num_time_markers'] * 5
        
        # Actions per minute
        features['actions_per_minute'] = features['num_actions'] / (features['session_duration'] / 60 + 0.001)
        
        # ---- ACTION TYPE FEATURES ----
        print("  - Action type frequency features...")
        
        # Extract all unique action types
        all_actions = []
        for idx, row in df.iterrows():
            actions = self.extract_session_actions(row, start_col)
            filtered = [self.filter_action(a) for a in actions if self.filter_action(a)]
            all_actions.extend(filtered)
        
        unique_actions = list(set(all_actions))
        print(f"    Found {len(unique_actions)} unique action types")
        
        # Count each action type per session
        for action in unique_actions[:50]:  # Limit to top 50 actions to avoid too many features
            features[f'action_{action}'] = df.iloc[:, start_col:].apply(
                lambda row: sum(1 for val in row if pd.notna(val) and 
                               self.filter_action(str(val)) == action), axis=1
            )
        
        # Action diversity (unique actions per session)
        features['action_diversity'] = df.iloc[:, start_col:].apply(
            lambda row: len(set(self.filter_action(str(val)) 
                               for val in row if pd.notna(val) and not str(val).startswith('t'))), 
            axis=1
        )
        
        # ---- SCREEN AND CONFIGURATION FEATURES ----
        print("  - Screen and configuration features...")
        
        # Most used screen
        features['most_used_screen'] = df.iloc[:, start_col:].apply(
            self._extract_most_common_pattern, pattern=self.pattern_ecran, axis=1
        )
        
        # Most used configuration
        features['most_used_config'] = df.iloc[:, start_col:].apply(
            self._extract_most_common_pattern, pattern=self.pattern_conf_ecran, axis=1
        )
        
        # Most used category (chaine)
        features['most_used_category'] = df.iloc[:, start_col:].apply(
            self._extract_most_common_pattern, pattern=self.pattern_chaine, axis=1
        )
        
        # ---- TEMPORAL PATTERNS ----
        print("  - Temporal pattern features...")
        
        # Average time between actions
        features['avg_time_between_actions'] = (
            features['session_duration'] / (features['num_actions'] + 1)
        )
        
        # ---- BROWSER FEATURES ----
        print("  - Browser encoding...")
        
        browser_col = 'navigateur'
        # One-hot encode browser
        browser_dummies = pd.get_dummies(df[browser_col], prefix='browser')
        
        # Combine all features
        features_df = pd.DataFrame(features)
        features_df = pd.concat([features_df, browser_dummies], axis=1)
        
        # Add target if training
        if is_train and 'util' in df.columns:
            features_df.insert(0, 'util', df['util'])
        
        print(f"\nFeature extraction complete! Shape: {features_df.shape}")
        return features_df
    
    def _extract_most_common_pattern(self, row, pattern):
        """Extract most common pattern match from row"""
        matches = []
        for val in row:
            if pd.notna(val):
                found = pattern.findall(str(val))
                matches.extend(found)
        
        if matches:
            counter = Counter(matches)
            return counter.most_common(1)[0][0]
        return 'none'