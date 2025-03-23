import torch
import yaml
from neuralop.models.codano import CODANO

def load_codano_config(config_path="config/darcy_config_codano.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    codano_config = config.get("codano", {})
    in_channels = codano_config.get("data_channels", 1)
    output_variable_codimension = codano_config.get("output_variable_codimension", 1)
    hidden_variable_codimension = codano_config.get("hidden_variable_codimension", 2)
    lifting_channels = codano_config.get("lifting_channels", 4)
    n_modes = codano_config.get("n_modes", [[16, 16]] * 4)
    n_heads = codano_config.get("n_heads", [1] * 4)
    n_layers = codano_config.get("n_layers", 4)
    per_layer_scaling_factors = codano_config.get("per_layer_scaling_factors", [[1, 1] for _ in range(n_layers)])
    attention_scaling_factors = codano_config.get("attention_scaling_factors", [1] * n_layers)
    return in_channels, output_variable_codimension, hidden_variable_codimension, lifting_channels, n_modes, n_heads, per_layer_scaling_factors, attention_scaling_factors, n_layers
def main():
    (in_channels, output_variable_codimension, hidden_variable_codimension, 
     lifting_channels, n_modes, n_heads, per_layer_scaling_factors, 
     attention_scaling_factors, n_layers) = load_codano_config()
    model = CODANO(
        in_channels=in_channels,
        output_variable_codimension=output_variable_codimension,
        hidden_variable_codimension=hidden_variable_codimension,
        lifting_channels=lifting_channels,
        n_modes=n_modes,
        n_heads=n_heads,
        per_layer_scaling_factors=per_layer_scaling_factors,
        attention_scaling_factors=attention_scaling_factors,
        n_layers=n_layers
    )
    weight_path = "models/model-codano-darcy-16-resolution-2025-03-15-19-31.pt"
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    print("CODANO loaded successfully!")
    
    dummy_input = torch.randn(1, in_channels, 16, 16)
    output = model(dummy_input)
    print("Output:", output)



if __name__ == "__main__":
    main()
