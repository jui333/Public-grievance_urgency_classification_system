from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from backend.model_service import train_model


def main():
    print("\n" + "="*60)
    print("🚀 Starting Model Training...")
    print("="*60 + "\n")
    
    model, results = train_model()
    
    model_path = project_root / "backend" / "category_model.joblib"
    print("\n" + "="*60)
    print(f"✅ Saved trained model to {model_path}")
    print("="*60 + "\n")
    print("Sample predictions:")
    print(results.head())


if __name__ == "__main__":
    main()
