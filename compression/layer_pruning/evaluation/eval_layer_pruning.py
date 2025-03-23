import torch
import torch.nn as nn
from compression.layer_pruning.layer_pruning import GlobalLayerPruning

def evaluate_model(model, dummy_input):
    """
    Run the model in evaluation mode on the provided input.
    """
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    return output

def main():
    # Define a simple model for evaluation.
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    # Instantiate the model and create dummy input data.
    model = SimpleModel()
    print("Original model:")
    print(model)

    dummy_input = torch.randn(1, 10)
    original_output = evaluate_model(model, dummy_input)
    print("Original output:", original_output)

    # Apply layer pruning with a 50% prune ratio.
    pruner = GlobalLayerPruning(model)
    pruner.layer_prune(prune_ratio=0.5)
    
    print("\nPruned model:")
    print(model)
    pruned_output = evaluate_model(model, dummy_input)
    print("Pruned output:", pruned_output)

if __name__ == "__main__":
    main()
