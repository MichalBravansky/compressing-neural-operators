import wandb

wandb.login()

models_dir = "models/" 
artifacts = [
    # Darcy 16 FNO
    #"ucl-neural-operator/training/model-fno-darcy-16-resolution-2025-03-04-18-48:v0",
    # Darcy 32 FNO
    #"ucl-neural-operator/training/model-fno-darcy-16-resolution-2025-03-17-19-02:v0",
    # Darcy 128 FNO
    #"ucl-neural-operator/training/model-fno-darcy-16-resolution-2025-03-17-18-57:v0",

    # Codano 128x128
    #"ucl-neural-operator/training/model-codano-darcy-16-resolution-2025-03-15-19-31:v0",

    # DeepONet 128x128
    "ucl-neural-operator/training/model-deeponet-darcy-128-resolution-2025-03-04-18-53:v0",

    # Foundational Neural Operators
    #'ucl-neural-operator/data/foundational-codano-full-run-all-weights:v0', # foundational codano weights
    #'ucl-neural-operator/data/foundational-fno-full-run:v0' # foundational fno weights

]

run = wandb.init()


for artifact_name in artifacts:
    #artifact = run.use_artifact(artifact_name, type="model")
    artifact = run.use_artifact(artifact_name, type="model")
    artifact_dir = artifact.download(root=models_dir)
    print(f"Downloaded {artifact_name} to {artifact_dir}")

run.finish()