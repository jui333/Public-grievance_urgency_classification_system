from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from backend.model_service import train_model


def main():
    model, results = train_model()
    model_path = project_root / "backend" / "category_model.joblib"
    print(f"Saved trained model to {model_path}")
    print("Sample predictions:")
    print(results.head())


if __name__ == "__main__":
    main()
