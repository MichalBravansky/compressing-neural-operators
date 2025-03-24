import wandb

wandb.login()

models_dir = "neuralop/data/datasets/data" 
artifacts = [
    'ucl-neural-operator/data/foundational-fno-data:v0',  # foundational fno dataset
    'ucl-neural-operator/data/foundational-codano-full-data:v0', # foundational codano dataset

]

run = wandb.init()

for artifact_name in artifacts:
    artifact = run.use_artifact(artifact_name, type="dataset")
    artifact_dir = artifact.download(root=models_dir)
    print(f"Downloaded {artifact_name} to {artifact_dir}")

run.finish()
