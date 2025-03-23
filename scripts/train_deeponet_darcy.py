import torch
import wandb
import sys
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from neuralop.training import setup, AdamW
from neuralop import get_model
from neuralop.utils import get_wandb_api_key
from neuralop.losses.data_losses import LpLoss, H1Loss
from neuralop.training.trainer import Trainer
from neuralop.data.datasets import DarcyDataset
from torch.utils.data import DataLoader, DistributedSampler
from neuralop.utils import get_wandb_api_key, count_model_params
from neuralop.mpu.comm import get_local_rank
from neuralop.data.datasets import load_darcy_flow_small
from torch.nn import functional as F

# Read the configuration
config_name = "default"
pipe = ConfigPipeline([
    YamlConfig("./deeponet_darcy_config.yaml", config_name=config_name, config_folder="./config"),
    ArgparseConfig(infer_types=True, config_name=None, config_file=None),
    YamlConfig(config_folder="./config"),
])
config = pipe.read_conf()

# Set-up distributed communication, if using
device, is_logger = setup(config)

# Set up WandB logging
wandb_init_args = {}
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
    
    if config.wandb.name:
        wandb_name = f"{config.wandb.name}-{timestamp}"
    else:
        wandb_name = "_".join(
            f"{var}" for var in [
                "deeponet-darcy",
                config.data.train_resolution,
                timestamp
            ]
        )

    wandb_init_args = dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity
    )

    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.init(**wandb_init_args)

# Loading the Darcy flow dataset
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    train_resolution = config.data.train_resolution,
    n_train=config.data.n_train,
    batch_size=config.data.batch_size,
    test_resolutions=config.data.test_resolutions,
    n_tests=config.data.n_tests,
    test_batch_sizes=config.data.test_batch_sizes,
    encode_input=False,
    encode_output=False,
)

print(test_loaders[config.data.test_resolutions[0]].dataset[0]["x"].shape)

model = get_model(config)
model = model.to(device)

# Print model size
if config.verbose and is_logger:
    print(f'Model parameters: {count_model_params(model)}')

# Create optimizer
optimizer = AdamW(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay
)

# Setup scheduler
if config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        mode="min"
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.opt.scheduler_T_max
    )
elif config.opt.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
else:
    raise ValueError(f"Got scheduler={config.opt.scheduler}")

# Setup loss functions
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

if config.opt.training_loss == "l2":
    train_loss = l2loss
else:
    raise ValueError(f'Got training_loss={config.opt.training_loss} but expected "l2"')

eval_losses = {"l2": l2loss, "h1": h1loss}

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print config to screen
if config.verbose and is_logger:
    pipe.log()
    sys.stdout.flush()

# Reconfigure DataLoaders to use a DistributedSampler 
# if in distributed data parallel mode
if config.distributed.use_distributed:
    train_db = train_loader.dataset
    train_sampler = DistributedSampler(train_db, rank=get_local_rank())
    train_loader = DataLoader(dataset=train_db,
                              batch_size=config.data.batch_size,
                              sampler=train_sampler)
    for (res, loader), batch_size in zip(test_loaders.items(), config.data.test_batch_sizes):
        test_db = loader.dataset
        test_sampler = DistributedSampler(test_db, rank=get_local_rank())
        test_loaders[res] = DataLoader(dataset=test_db,
                              batch_size=batch_size,
                              shuffle=False,
                              sampler=test_sampler)

# Create trainer
trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    device=device,
    data_processor=data_processor,
    mixed_precision=config.opt.amp_autocast,
    wandb_log=config.wandb.log,
    eval_interval=config.wandb.eval_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose and is_logger,
)

# Log parameter count
if is_logger:
    n_params = count_model_params(model)

    if config.verbose:
        print(f"\nn_params: {n_params}")
        sys.stdout.flush()

    if config.wandb.log:
        to_log = {"n_params": n_params}
        wandb.log(to_log, commit=False)
        wandb.watch(model)

# Train the model
trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    training_loss=train_loss,
    eval_losses=eval_losses,
    regularizer=None,
)

if config.verbose and is_logger:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()

if config.wandb.log and is_logger:
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
    model_name = f"model-{config.wandb.name}-{timestamp}"
    
    torch.save(model.state_dict(), f"{model_name}.pt")

    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        description="DeepONet Darcy model"
    )
    artifact.add_file(f"{model_name}.pt")
    wandb.log_artifact(artifact)

    wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"))
    
    wandb.finish() 