import pickle
import traceback

model_files = {
    "coronary": "final_models/best_coronary_xgboost_dedicated.pkl",
    "heart_attack": "final_models/best_heartattack_model.pkl",
    "heart_failure": "final_models/best_heartfailure_model.pkl",
    "hypertension": "final_models/hypertension_model_compressed.pkl",
    "normal_heart": "final_models/normal_heart.pkl"
}

for model_name, model_path in model_files.items():
    try:
        print(f"\nLoading {model_name} from {model_path}...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Successfully loaded {model_name}")
        print(f"  Model type: {type(model)}")
    except Exception as e:
        print(f"✗ Failed to load {model_name}")
        print(f"  Error: {str(e)}")
        traceback.print_exc()
