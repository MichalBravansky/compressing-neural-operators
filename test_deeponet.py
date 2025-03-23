import torch
import yaml
from neuralop.models.deeponet import DeepONet
from ruamel.yaml.comments import CommentedSeq  # if needed

def load_deeponet_config(config_path="config/deeponet_darcy_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    deeponet_config = config.get("deeponet", {})
    train_resolution = deeponet_config.get("train_resolution", 128)
    in_channels = deeponet_config.get("in_channels", 1)
    out_channels = deeponet_config.get("out_channels", 1)
    hidden_channels = deeponet_config.get("hidden_channels", 64)
    branch_layers = deeponet_config.get("branch_layers", [256,256,256,256,128])
    trunk_layers = deeponet_config.get("trunk_layers", [256,256,256,256,128])
    return train_resolution, in_channels, out_channels, hidden_channels, branch_layers, trunk_layers

def main():
    train_resolution, in_channels, out_channels, hidden_channels, branch_layers, trunk_layers = load_deeponet_config()
    # Instantiate DeepONet with the configuration parameters.
    model = DeepONet(train_resolution, in_channels, out_channels, hidden_channels, branch_layers, trunk_layers)
    weight_path = "models/model-deeponet-darcy-128-resolution-2025-03-04-18-53.pt"
    with torch.serialization.safe_globals([]):
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    print("DeepONet loaded successfully!")
    
    # Create dummy inputs.
    # Provide a 4D tensor for the branch input.
    dummy_branch = torch.randn(1, in_channels, train_resolution, train_resolution)
    # For the trunk input, check your DeepONet implementation.
    # If it also expects a 4D tensor, do similarly; if it's 2D (e.g. coordinates), adjust accordingly.
    # For this example, we'll assume it's also a 4D tensor.
    dummy_trunk = torch.randn(1, in_channels, train_resolution, train_resolution)
    
    output = model(dummy_branch, dummy_trunk)
    if isinstance(output, (list, tuple)):
        print("Output shapes:", [o.shape for o in output])
    else:
        print("Output shape:", output.shape)

if __name__ == "__main__":
    main()
