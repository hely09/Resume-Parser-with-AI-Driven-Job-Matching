"""
Model Trainer - Trains ML models for category prediction
"""
import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from src.config import *

warnings.filterwarnings('ignore')


class ModelTrainer:
    """Train and compare multiple models for category prediction"""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.category_skills_map = {}
        self.label_encoder = LabelEncoder()
        self.vectorizer = None
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0
        self.results = {}

        # Initialize models
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
            'SVM': SVC(kernel='linear', probability=True, random_state=RANDOM_STATE),
            'KNN': KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss')
        }

    def load_and_prepare_data(self):
        """Load dataset and create category-skills mapping"""
        print("=" * 70)
        print("LOADING AND PREPARING DATA")
        print("=" * 70)

        self.df = pd.read_csv(self.csv_path)
        print(f"\n✓ Loaded {len(self.df)} records")
        print(f"✓ Categories: {self.df['category'].nunique()}")

        # Split comma-separated skills
        print("\n→ Splitting comma-separated skills...")
        expanded_rows = []
        for _, row in self.df.iterrows():
            category = row['category']
            skills_str = str(row['skill'])
            individual_skills = [s.strip().lower() for s in skills_str.split(',')]

            for skill in individual_skills:
                if skill:
                    expanded_rows.append({'category': category, 'skill': skill})

        self.df = pd.DataFrame(expanded_rows)
        print(f"✓ Expanded to {len(self.df)} individual skill entries")
        print(f"✓ Unique Skills: {self.df['skill'].nunique()}")

        # Create category-skills mapping
        print("\n→ Creating category-skills mapping...")
        for category in self.df['category'].unique():
            skills = self.df[self.df['category'] == category]['skill'].tolist()
            self.category_skills_map[category] = sorted(list(set(skills)))

        print(f"✓ Mapped {len(self.category_skills_map)} categories")

        # Display distribution
        print("\nCategory Distribution:")
        print("-" * 70)
        cat_counts = self.df['category'].value_counts()
        for cat, count in cat_counts.items():
            print(f"  {cat}: {count} skills")

        return self.df

    def prepare_features(self, vectorizer_type='tfidf'):
        """Prepare features for training"""
        print(f"\n→ Preparing features using {vectorizer_type.upper()}...")

        X = self.df['skill'].str.lower()
        y = self.df['category']

        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=MAX_FEATURES,
                ngram_range=NGRAM_RANGE,
                min_df=1
            )

        X_vec = self.vectorizer.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)

        print(f"✓ Feature matrix shape: {X_vec.shape}")
        print(f"✓ Training samples: {X_vec.shape[0]}")

        return X_vec, y_encoded

    def train_and_evaluate_models(self, X, y, test_size=TEST_SIZE):
        """Train multiple models and compare performance"""
        print("\n" + "=" * 70)
        print("TRAINING AND EVALUATING MODELS")
        print("=" * 70)

        n_samples = X.shape[0]
        print(f"\nTotal samples: {n_samples}")

        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
            )
        except ValueError:
            print("⚠ Cannot use stratified split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=RANDOM_STATE
            )

        print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

        results = []

        for model_name, model in self.models.items():
            print(f"\n{'-' * 70}")
            print(f"Training: {model_name}")
            print(f"{'-' * 70}")

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='weighted', zero_division=0
                )

                cv_folds = min(5, n_samples // 10) if n_samples >= 20 else 3
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
                cv_mean = cv_scores.mean()

                result = {
                    'Model': model_name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'CV Score': cv_mean
                }

                results.append(result)

                print(f"✓ Accuracy: {accuracy:.4f}")
                print(f"✓ F1-Score: {f1:.4f}")
                print(f"✓ CV Score: {cv_mean:.4f}")

                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_model = model
                    self.best_model_name = model_name

            except Exception as e:
                print(f"❌ Error: {str(e)}")
                continue

        self.results_df = pd.DataFrame(results)
        self.results_df = self.results_df.sort_values('Accuracy', ascending=False)

        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        print(self.results_df.to_string(index=False))

        print(f"\n{'=' * 70}")
        print(f"🏆 BEST MODEL: {self.best_model_name}")
        print(f"🏆 ACCURACY: {self.best_accuracy:.4f}")
        print(f"{'=' * 70}")

        return self.results_df

    def save_models(self):
        """Save trained models and components"""
        print("\n→ Saving models...")

        with open(TRAINED_MODEL_PATH, 'wb') as f:
            pickle.dump(self.best_model, f)

        with open(VECTORIZER_PATH, 'wb') as f:
            pickle.dump(self.vectorizer, f)

        with open(LABEL_ENCODER_PATH, 'wb') as f:
            pickle.dump(self.label_encoder, f)

        with open(CATEGORY_SKILLS_MAP_PATH, 'wb') as f:
            pickle.dump(self.category_skills_map, f)

        print(f"✓ Models saved to {MODELS_DIR}")


def load_trained_models():
    """Load pre-trained models"""
    with open(TRAINED_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)

    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)

    with open(CATEGORY_SKILLS_MAP_PATH, 'rb') as f:
        category_skills_map = pickle.load(f)

    return {
        'model': model,
        'vectorizer': vectorizer,
        'label_encoder': label_encoder,
        'category_skills_map': category_skills_map
    }

