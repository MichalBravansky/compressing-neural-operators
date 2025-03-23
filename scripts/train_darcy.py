import sys
import os
import torch
import torch.nn as nn
import yaml
import copy

# --- Adjust sys.path so that the repository root is found ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# --- Import our pruning modules ---
from compression.layer_pruning.layer_pruning import GlobalLayerPruning
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
# Import the real evaluation routines from utils.py:
from compression.utils import compare_models

# --- Import the actual models ---
from neuralop.models.fno import FNO
from neuralop.models.deeponet import DeepONet
from neuralop.models.gino import GINO
from neuralop.models.codano import CODANO

# --- Import the Darcy data loader (correct version) ---
from neuralop.data.datasets.darcy import load_darcy_flow_small

# --- FNO configuration loader (assumes keys in your YAML file) ---
def load_fno_config(config_path="config/darcy_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return (
        config.get("n_modes", (16, 16)),
        config.get("in_channels", 1),
        config.get("out_channels", 1),
        config.get("hidden_channels", 32)
    )

# --- Load the Darcy dataset ---
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000,
    batch_size=16,
    test_resolutions=[16, 32],
    n_tests=[100, 50],
    test_batch_sizes=[16, 16],
    encode_input=False, 
    encode_output=False,
)

# --- Updated load_and_prune_model using compare_models from utils.py ---
def load_and_prune_model(ModelClass, weight_path, test_loaders, data_processor, device, prune_ratio=0.2, technique="layer"):
    """
    Loads a model from weight_path, creates a deep copy for pruning,
    applies the selected pruning technique (either "layer" or "magnitude"),
    and then evaluates both versions using compare_models.
    """
    # Handle FNO instantiation (which requires additional parameters)
    if ModelClass.__name__ == "FNO":
        n_modes, in_channels, out_channels, hidden_channels = load_fno_config()
        base_model = ModelClass(n_modes, in_channels, out_channels, hidden_channels)
    else:
        base_model = ModelClass()
    
    # Load model weights onto the base model.
    base_model.load_state_dict(torch.load(weight_path, map_location=device))
    base_model = base_model.to(device)
    base_model.eval()
    
    # Create a deep copy to apply pruning on.
    pruned_model = copy.deepcopy(base_model)
    
    # Apply the chosen pruning technique.
    if technique == "layer":
        pruner = GlobalLayerPruning(pruned_model)
        pruner.layer_prune(prune_ratio=prune_ratio)
    elif technique == "magnitude":
        pruner = GlobalMagnitudePruning(pruned_model, prune_ratio=prune_ratio)
        pruner.prune()  # Adjust if the API differs.
    else:
        raise ValueError("Unknown compression technique: choose 'layer' or 'magnitude'")
    
    # Use the real evaluation routine to compare models.
    results = compare_models(
        model1=base_model,
        model2=pruned_model,
        test_loaders=test_loaders,
        data_processor=data_processor,
        device=device,
        verbose=True,
        track_performance=True
    )
    
    return base_model, pruned_model, results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set the compression technique you want to evaluate: "layer" or "magnitude"
    technique = "layer"
    
    # --- Section 1: Evaluate all 4 models ---
    # Replace these placeholder paths with the actual model weight file paths.
    models_info = {
        "FNO": {"class": FNO, "weight": "models/model-fno-darcy-16-resolution-2025-02-05-19-55.pt"},
        "DeepONet": {"class": DeepONet, "weight": "models/model-deeponet-darcy-16-2025-02-XX.pt"},
        "GINO": {"class": GINO, "weight": "models/model-gino-carcfd-32-resolution-2025-02-12-18-47.pt"},
        "Codano": {"class": CODANO, "weight": "models/model-codano-darcy-16-resolution-2025-02-11-21-13.pt"},
    }
    
    results_section1 = {}
    print("=== Section 1: Evaluating all 4 models using {} pruning ===".format(technique))
    for model_name, info in models_info.items():
        print(f"\nProcessing {model_name}...")
        model_class = info["class"]
        weight_path = info["weight"]
        try:
            base_model, pruned_model, results = load_and_prune_model(
                model_class, weight_path, test_loaders, data_processor, device, prune_ratio=0.2, technique=technique
            )
            results_section1[model_name] = results
            print(f"{model_name} evaluation results:")
            print(results)
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
    
    # --- Section 2: Compare 'Our' vs 'Their' weights for FNO and Codano ---
    models_info_section2 = {
        "FNO": {
            "our": "models/model-our_fno_darcy-16-2025-XX.pt",
            "theirs": "models/model-their_fno_darcy-16-2025-XX.pt"
        },
        "Codano": {
            "our": "models/model-our_codano_darcy-16-2025-XX.pt",
            "theirs": "models/model-their_codano_darcy-16-2025-XX.pt"
        }
    }
    
    results_section2 = {}
    print("\n=== Section 2: Comparing 'Our' vs 'Their' weights for FNO and Codano using {} pruning ===".format(technique))
    for model_name, weight_paths in models_info_section2.items():
        results_section2[model_name] = {}
        ModelClass = FNO if model_name == "FNO" else CODANO
        
        for variant, weight_path in weight_paths.items():
            print(f"\nProcessing {model_name} ({variant})...")
            try:
                base_model, pruned_model, results = load_and_prune_model(
                    ModelClass, weight_path, test_loaders, data_processor, device, prune_ratio=0.2, technique=technique
                )
                results_section2[model_name][variant] = results
                print(f"{model_name} ({variant}) evaluation results:")
                print(results)
            except Exception as e:
                print(f"Error processing {model_name} ({variant}): {e}")
    
    # Optionally, log or save the results for your report.
    print("\nSection 1 Results:")
    print(results_section1)
    print("\nSection 2 Results:")
    print(results_section2)

if __name__ == "__main__":
    main()
