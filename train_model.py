"""
Model Training Script
Run this once to train and save models before using the app
"""
from src.model_trainer import ModelTrainer
from src.config import IT_SKILLS_CSV


def main():
    print("="*70)
    print("RESUME ANALYZER - MODEL TRAINING")
    print("="*70)

    # Initialize trainer
    trainer = ModelTrainer(str(IT_SKILLS_CSV))

    # Load and prepare data
    trainer.load_and_prepare_data()

    # Prepare features
    X, y = trainer.prepare_features(vectorizer_type='tfidf')

    # Train models
    results = trainer.train_and_evaluate_models(X, y)

    # Save best model
    trainer.save_models()

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)
    print("\nYou can now run the Streamlit app:")
    print("  streamlit run app.py")


if __name__ == '__main__':
    main()