"""
Configuration file for Resume Analyzer
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, UPLOADS_DIR, VECTOR_DB_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset paths
IT_SKILLS_CSV = DATA_DIR / "synthetic_it_skills_dataset.csv"
COURSES_CSV = DATA_DIR / "Coursera_courses.csv"

# Model paths
TRAINED_MODEL_PATH = MODELS_DIR / "trained_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
CATEGORY_SKILLS_MAP_PATH = MODELS_DIR / "category_skills_map.pkl"

# Semantic model configuration
SEMANTIC_MODEL_NAME = 'all-MiniLM-L6-v2'

# Skill matching thresholds
SEMANTIC_THRESHOLD = 0.75
MIN_SKILL_LENGTH = 2
MAX_SKILL_LENGTH = 50

# Model training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 500
NGRAM_RANGE = (1, 2)

# Course recommendation parameters
TOP_N_COURSES_PER_SKILL = 2
MAX_SKILLS_FOR_COURSES = 10

# Streamlit configuration
PAGE_TITLE = "AI Resume Analyzer"
PAGE_ICON = "📄"
LAYOUT = "wide"

# Color scheme
PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"
SUCCESS_COLOR = "#2ca02c"
WARNING_COLOR = "#d62728"