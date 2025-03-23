import torch
from neuralop.models.fno import FNO
from ruamel.yaml.scalarint import ScalarInt
from ruamel.yaml.scalarfloat import ScalarFloat

def main():
    # Use the training parameters from your config (FNO, from darcy_config.yaml)
    n_modes = (32, 32)
    in_channels = 1
    out_channels = 1
    hidden_channels = 64
    n_layers = 5

    # Instantiate FNO with the parameters matching training
    model = FNO(
        n_modes, in_channels, out_channels, hidden_channels,
        n_layers=n_layers, skip="linear", norm="group_norm",
        implementation="factorized", projection_channel_ratio=2,
        separable=False, dropout=0.0, rank=1.0
    )

    weight_path = "models/model-fno-darcy-16-resolution-2025-03-04-18-48.pt"
    # Wrap torch.load in the safe_globals context to allow custom globals.
    with torch.serialization.safe_globals([ScalarInt, ScalarFloat]):
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    print("FNO loaded successfully!")
    # Optionally, perform a dummy forward pass:
    dummy_input = torch.randn(1, in_channels, 128, 128)  # resolution must match training input
    output = model(dummy_input)
    print("Output shape:", output.shape if not isinstance(output, (list, tuple)) else [o.shape for o in output])

if __name__ == "__main__":
    main()
