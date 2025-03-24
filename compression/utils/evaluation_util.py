import torch
import time
from neuralop.losses import LpLoss, H1Loss
from neuralop.data.datasets import load_darcy_flow_small
from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info
import pprint
import torch.nn as nn
from timeit import default_timer
from tqdm import tqdm
import wandb
from neuralop.data_utils.data_utils import batched_masker
from neuralop.utils import prepare_input
from compression.utils.codano_util import get_grid_displacement



def evaluate_model(model, 
                   dataloader, 
                   data_processor=None, 
                   device='cuda', 
                   track_performance=False, 
                   verbose=False, 
                   evaluation_params=None):
    """
    Evaluates model performance with optional tracking of runtime, memory usage, and FLOPs,
    while processing inputs in the same manner as the `missing_variable_testing` function.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to be evaluated.
    dataloader : torch.utils.data.DataLoader
        The DataLoader for evaluation.
    data_processor : Module or None
        Optional data processor for any preprocessing/postprocessing.
    device : str
        The device on which to run the evaluation (default 'cuda').
    track_performance : bool
        Whether to track runtime, memory usage, and FLOPs (default False).
    verbose : bool
        Whether to print detailed information during evaluation (default False).
    evaluation_params : dict or None
        Dictionary containing optional parameters for evaluation:
          - params: A configuration object or dict that may have 'horizontal_skip' or other attributes.
          - stage: The stage of inference (used by get_grid_displacement).
          - variable_encoder, token_expander, input_mesh: 
                for preparing the input via `prepare_input`.
          - augmenter: If present, applies input masking via `batched_masker`.

    Returns
    -------
    dict
        Dictionary containing:
            'l2_loss': Average L2 loss across the dataset (using LpLoss with p=2).
            'h1_loss': Average H1 loss across the dataset (using H1Loss).
            If track_performance=True, also includes:
            'runtime', 'model_size_mb', 'peak_memory_mb', and 'flops' (if available).
    """
    model.eval()
    
    # Metrics we keep from the original evaluate_model
    l2_loss_fn = LpLoss(d=2, p=2, reduction='mean')
    h1_loss_fn = H1Loss(d=2, reduction='mean')
    loss_p = nn.MSELoss()

    total_l2_loss = 0.0
    total_h1_loss = 0.0
    total_runtime = 0.0
    batch_count = 0
    
    flops_counted = False
    flops = 0
    model_size_mb = 0
    
    if track_performance and torch.cuda.is_available():
        # Move to CPU first to measure memory usage for load
        model = model.to('cpu')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        before_model_load = torch.cuda.memory_allocated(device)

        model = model.to(device)
        model_size_mb = (torch.cuda.memory_allocated(device) - before_model_load) / (1024 * 1024)

        torch.cuda.reset_peak_memory_stats(device)
        start_memory = torch.cuda.memory_allocated(device)
    else:
        model = model.to(device)

    if data_processor is not None:
        data_processor = data_processor.to(device)
        data_processor.eval()

    # Extract anything we need from evaluation_params
    if evaluation_params is not None:
        params = evaluation_params.get('params', None)
        stage = evaluation_params.get('stage', None)
        variable_encoder = evaluation_params.get('variable_encoder', None)
        token_expander = evaluation_params.get('token_expander', None)
        initial_mesh = evaluation_params.get('input_mesh', None)
        augmenter = evaluation_params.get('augmenter', None)
    else:
        params = None
        stage = None
        variable_encoder = None
        token_expander = None
        initial_mesh = None
        augmenter = None
        
    with torch.no_grad():
        for batch in tqdm(dataloader, disable=False): # show the bar
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            static_features = batch.get("static_features", None)
            if augmenter is not None:
                x, _ = batched_masker(x, augmenter)

            if data_processor is not None:
                batch_processed = data_processor.preprocess(batch)
            else:
                batch_processed = {k: v.to(device) 
                                   for k, v in batch.items() if torch.is_tensor(v)}

            if params is not None and stage is not None:
                inp = prepare_input(
                    x,
                    static_features,
                    params,
                    variable_encoder,
                    token_expander,
                    initial_mesh,
                    batch
                )
                # Also get the displacements
                out_grid_displacement, in_grid_displacement = get_grid_displacement(params, stage, batch)

                # Overwrite or build a dictionary to pass to the model
                model_input = {
                    "x": inp,
                    "y": y,  # not used in forward, but let's keep for clarity
                    "out_grid_displacement": out_grid_displacement,
                    "in_grid_displacement": in_grid_displacement
                }
            else:
                # If no special params/stage, we just use the original approach
                model_input = batch_processed
                # Make sure 'x' and 'y' are present
                model_input["x"] = x
                model_input["y"] = y

            # -- FLOPS measurement on the first batch if requested --
            if track_performance and not flops_counted:
                try:
                    from ptflops import get_model_complexity_info

                    def input_constructor(_):
                        # Make a clone of the dictionary so that we don't mess up references
                        # we only pass the TENSOR inputs needed by the model
                        # (ptflops can have issues with non-tensor inputs)
                        clone_dict = {}
                        for mk, mv in model_input.items():
                            if torch.is_tensor(mv):
                                clone_dict[mk] = mv.clone()
                            
                        if 'x' in clone_dict and len(clone_dict['x'].shape) >= 2:
                            is_codano = False
                            
                            if hasattr(model, 'model'):
                                inner_model = model.model
                                is_codano = (hasattr(inner_model, '__class__') and 
                                             'codano' in str(inner_model.__class__.__name__).lower())
                            elif 'codano' in str(model.__class__.__name__).lower():
                                is_codano = True
                                
                            if is_codano and 'variable_ids' not in clone_dict:
                                clone_dict['variable_ids'] = ["a1"]
                                
                            
                        if 'x' in clone_dict and len(clone_dict['x'].shape) >= 2:
                            is_codano = False
                            
                            if hasattr(model, 'model'):
                                inner_model = model.model
                                is_codano = (hasattr(inner_model, '__class__') and 
                                             'codano' in str(inner_model.__class__.__name__).lower())
                            elif 'codano' in str(model.__class__.__name__).lower():
                                is_codano = True
                                
                            if is_codano and 'variable_ids' not in clone_dict:
                                clone_dict['variable_ids'] = ["a1"]
                                
                        return clone_dict

                    macs, params_model = get_model_complexity_info(
                        model, 
                        (1,),  # dummy input shape, will be overridden by input_constructor
                        input_constructor=input_constructor,
                        as_strings=False, 
                        print_per_layer_stat=False, 
                        verbose=verbose,
                        backend='aten'
                    )
                    flops = macs * 2  # MACs -> FLOPs
                    flops_counted = True
                except Exception as e:
                    if verbose:
                        print(f"Error measuring FLOPs with ptflops: {e}")
                    flops = 0

            # -- Runtime measurement --
            start_time = time.time() if track_performance else None

            # Forward pass
            out = model(**model_input)
            
            # Horizontal skip connection if specified
            if params is not None and getattr(params, 'horizontal_skip', False):
                # The shape of 'out' must match 'x' here
                out = out + x

            # End runtime measurement
            if track_performance:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                total_runtime += (time.time() - start_time)

            # If we have a data_processor, postprocess the output
            if data_processor is not None:
                out, model_input = data_processor.postprocess(out, model_input)

            target = model_input['y']

            if evaluation_params is not None:
                batch_size = out.shape[0]
                total_l2_loss += loss_p(out.reshape(batch_size, -1), 
                                        target.reshape(batch_size, -1)).item()
            else:
                total_l2_loss += l2_loss_fn(out, target).item()
                total_h1_loss += h1_loss_fn(out, target).item()
            
            batch_count += 1

    # Averages
    avg_l2_loss = total_l2_loss / max(batch_count, 1)
    avg_h1_loss = total_h1_loss / max(batch_count, 1)

    result = {
        'l2_loss': avg_l2_loss,
        'h1_loss': avg_h1_loss,
    }

    # Performance stats
    if track_performance:
        result['runtime'] = total_runtime / max(batch_count, 1)

        if model_size_mb > 0:
            result['model_size_mb'] = model_size_mb

        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated(device) - start_memory
            result['peak_memory_mb'] = peak_memory / (1024 * 1024)

        if flops > 0:
            result['flops'] = flops

    return result

def compare_models(model1, model2, test_loaders, data_processor, device, 
                  model1_name="Original Model", model2_name="Compressed Model",
                  verbose=True, track_performance=False, evaluation_params=None):
    """Compare performance between two models across different resolutions.
    
    Args:
        model1: First model to evaluate (e.g., original model)
        model2: Second model to evaluate (e.g., compressed model)
        test_loaders: Dict of test loaders for different resolutions
        data_processor: Data processor for the dataset
        device: Device to run evaluation on
        model1_name: Name for the first model (default: "Original Model")
        model2_name: Name for the second model (default: "Compressed Model")
        verbose: Whether to print detailed results (default: True)
        track_performance: Whether to track runtime, memory usage and FLOPs (default: False)
    """
    results = {}
    
    if verbose:
        print("\n" + "="*50)
        print(f"{model1_name.upper()} EVALUATION")
        print("="*50)
    
    for resolution, loader in test_loaders.items():
        if verbose:
            print(f"\nResults on {resolution}x{resolution} resolution")
            print("-"*30)
        results[f"{resolution}_base"] = evaluate_model(model1, loader, data_processor, device, 
                                                     track_performance=track_performance, 
                                                     evaluation_params=evaluation_params)
        if verbose:
            print(f"L2 Loss: {results[f'{resolution}_base']['l2_loss']:.6f}")
            if track_performance:
                if 'runtime' in results[f'{resolution}_base']:
                    print(f"Avg Runtime per batch: {results[f'{resolution}_base']['runtime']*1000:.2f} ms")
                if 'model_size_mb' in results[f'{resolution}_base']:
                    print(f"Model Size: {results[f'{resolution}_base']['model_size_mb']:.2f} MB")
                if 'peak_memory_mb' in results[f'{resolution}_base']:
                    print(f"Peak Memory Usage: {results[f'{resolution}_base']['peak_memory_mb']:.2f} MB")
                if 'flops' in results[f'{resolution}_base']:
                    print(f"FLOPs: {results[f'{resolution}_base']['flops']/1e9:.2f} GFLOPs")
    
    if verbose:
        print("\n" + "="*50)
        print(f"{model2_name.upper()} EVALUATION")
        print("="*50)
    
    if hasattr(model2, 'get_compression_stats') and verbose:
        stats = model2.get_compression_stats()
        print(f"\nModel sparsity: {stats['sparsity']:.2%}")

        if 'original_size' in stats:
            print(f"Original size: {stats['original_size']} bytes")
        if 'quantized_size' in stats:
            print(f"Quantized size: {stats['quantized_size']} bytes")
        if 'compression_ratio' in stats:
            print(f"Compression ratio: {stats['compression_ratio']:.2f}")
        if 'dyquantized_layers' in stats:
            print("Quantized layers:")
            for layer_name in stats['dyquantized_layers']:
                print(f" - {layer_name}")
    
    for resolution, loader in test_loaders.items():
        if verbose:
            print(f"\nResults on {resolution}x{resolution} resolution")
            print("-"*30)
        results[f"{resolution}_compressed"] = evaluate_model(model2, loader, data_processor, device,
                                                          track_performance=track_performance,
                                                          evaluation_params=evaluation_params)
        if verbose:
            print(f"L2 Loss: {results[f'{resolution}_compressed']['l2_loss']:.6f}")
            if track_performance:
                if 'runtime' in results[f'{resolution}_compressed']:
                    print(f"Avg Runtime per batch: {results[f'{resolution}_compressed']['runtime']*1000:.2f} ms")
                if 'model_size_mb' in results[f'{resolution}_compressed']:
                    print(f"Model Size: {results[f'{resolution}_compressed']['model_size_mb']:.2f} MB")
                if 'peak_memory_mb' in results[f'{resolution}_compressed']:
                    print(f"Peak Memory Usage: {results[f'{resolution}_compressed']['peak_memory_mb']:.2f} MB")
                if 'flops' in results[f'{resolution}_compressed']:
                    print(f"FLOPs: {results[f'{resolution}_compressed']['flops']/1e9:.2f} GFLOPs")
    
        if verbose:
            print("\n" + "="*50)
            print("PERFORMANCE COMPARISON")
            print("="*50)
            print("\nRelative increase in error (compressed vs original):")
            print("-"*50)
        results["Comparison"] = {}
        if verbose:
            print("\n" + "="*50)
            print("PERFORMANCE COMPARISON")
            print("="*50)
            print("\nRelative increase in error (compressed vs original):")
            print("-"*50)
        results["Comparison"] = {}
        for resolution in test_loaders.keys():
            base_results = results[f"{resolution}_base"]
            comp_results = results[f"{resolution}_compressed"]
            l2_change_percentage = (comp_results['l2_loss'] / base_results['l2_loss'] - 1) * 100
            results["Comparison"]["l2_loss_increase"] = l2_change_percentage
            print(f"{resolution}x{resolution} - L2: {l2_change_percentage:.2f}%")
            
            # Performance comparison if tracking enabled
            if track_performance:
                if 'runtime' in base_results and 'runtime' in comp_results:
                    if comp_results['runtime'] == 0:
                        speedup = 0
                    else:
                        speedup = (base_results['runtime'] / comp_results['runtime']) * 100
                    results["Comparison"]["run_time_speed_up"] = speedup
                    if comp_results['runtime'] == 0:
                        speedup = 0
                    else:
                        speedup = (base_results['runtime'] / comp_results['runtime']) * 100
                    results["Comparison"]["run_time_speed_up"] = speedup
                    print(f"{resolution}x{resolution} - Runtime Speedup: {speedup:.2f}x")
                
                if 'model_size_mb' in base_results and 'model_size_mb' in comp_results:
                    model_size_reduction = (1 - comp_results['model_size_mb'] / base_results['model_size_mb']) * 100
                    results["Comparison"]["model_size_reduction"] = model_size_reduction
                    print(f"{resolution}x{resolution} - Model Size Reduction: {model_size_reduction:.2f}%")
                
                if 'peak_memory_mb' in base_results and 'peak_memory_mb' in comp_results:
                    memory_reduction = (1 - comp_results['peak_memory_mb'] / base_results['peak_memory_mb']) * 100
                    results["Comparison"]["peak_memory_reduction"] = memory_reduction
                    results["Comparison"]["peak_memory_reduction"] = memory_reduction
                    print(f"{resolution}x{resolution} - Peak Memory Reduction: {memory_reduction:.2f}%")
                
                if 'flops' in base_results and 'flops' in comp_results:
                    flops_reduction = (1 - comp_results['flops'] / base_results['flops']) * 100
                    results["Comparison"]["flops_reduction"] = flops_reduction
                    print(f"{resolution}x{resolution} - FLOPs Reduction: {flops_reduction:.2f}%")
            
    
    return results



def compare_models_hyperparams(model1, model2s, hyperparameters, test_loaders, data_processor, device, 
                  model1_name="Original Model", model2_name="Compressed Model",
                  verbose=True, track_performance=False, evaluation_params=None):
    """Compare performance between two models across different resolutions.
    
    Args:
        model1: First model to evaluate (e.g., original model)
        model2: Second model to evaluate (e.g., compressed model)
        test_loaders: Dict of test loaders for different resolutions
        data_processor: Data processor for the dataset
        device: Device to run evaluation on
        model1_name: Name for the first model (default: "Original Model")
        model2_name: Name for the second model (default: "Compressed Model")
        verbose: Whether to print detailed results (default: True)
        track_performance: Whether to track runtime, memory usage and FLOPs (default: False)
    """
    results = {}
    results["Comparison"] = {}
    resolution, loader = next(iter(test_loaders.items()))
    results[f"{resolution}_compressed"] = {}
    if verbose:
        print("\n" + "="*50)
        print(f"{model1_name.upper()} EVALUATION")
        print("="*50)
    
    for resolution, loader in test_loaders.items():
        if verbose:
            print(f"\nResults on {resolution}x{resolution} resolution")
            print("-"*30)
        results[f"{resolution}_base"] = evaluate_model(model1, loader, data_processor, device, 
                                                     track_performance=track_performance, 
                                                     evaluation_params=evaluation_params)
        if verbose:
            print(f"L2 Loss: {results[f'{resolution}_base']['l2_loss']:.6f}")
            if track_performance:
                if 'runtime' in results[f'{resolution}_base']:
                    print(f"Avg Runtime per batch: {results[f'{resolution}_base']['runtime']*1000:.2f} ms")
                if 'model_size_mb' in results[f'{resolution}_base']:
                    print(f"Model Size: {results[f'{resolution}_base']['model_size_mb']:.2f} MB")
                if 'peak_memory_mb' in results[f'{resolution}_base']:
                    print(f"Peak Memory Usage: {results[f'{resolution}_base']['peak_memory_mb']:.2f} MB")
                if 'flops' in results[f'{resolution}_base']:
                    print(f"FLOPs: {results[f'{resolution}_base']['flops']/1e9:.2f} GFLOPs")
    
    for k in range(len(hyperparameters)):
        hyperparameter = hyperparameters[k]
        model2 = model2s[k]
        if verbose:
            print("\n")
            print("<"+"-"*50, f"{model2_name.upper()} EVALUATION with hyperparam({str(hyperparameter)})", 50*"-"+">")
        
        if hasattr(model2, 'get_compression_stats') and verbose:
            stats = model2.get_compression_stats()
            print(f"\nModel sparsity: {stats['sparsity']:.2%}")

            if 'original_size' in stats:
                print(f"Original size: {stats['original_size']} bytes")
            if 'quantized_size' in stats:
                print(f"Quantized size: {stats['quantized_size']} bytes")
            if 'compression_ratio' in stats:
                print(f"Compression ratio: {stats['compression_ratio']:.2f}")
            if 'dyquantized_layers' in stats:
                print("Quantized layers:")
                for layer_name in stats['dyquantized_layers']:
                    print(f" - {layer_name}")
    


        results[f"{resolution}_compressed"] = {}
        if verbose:
            print(f"\nResults on {resolution}x{resolution} resolution")
            print("-"*30)
        results[f"{resolution}_compressed"][hyperparameter] = evaluate_model(model2, loader, data_processor, device,
                                                        track_performance=track_performance,
                                                        evaluation_params=evaluation_params)
        if verbose:
            print(f"L2 Loss: {results[f'{resolution}_compressed'][hyperparameter]['l2_loss']:.6f}")
            if track_performance:
                if 'runtime' in results[f'{resolution}_compressed'][hyperparameter]:
                    print(f"Avg Runtime per batch: {results[f'{resolution}_compressed'][hyperparameter]['runtime']*1000:.2f} ms")
                if 'model_size_mb' in results[f'{resolution}_compressed'][hyperparameter]:
                    print(f"Model Size: {results[f'{resolution}_compressed'][hyperparameter]['model_size_mb']:.2f} MB")
                if 'peak_memory_mb' in results[f'{resolution}_compressed'][hyperparameter]:
                    print(f"Peak Memory Usage: {results[f'{resolution}_compressed'][hyperparameter]['peak_memory_mb']:.2f} MB")
                if 'flops' in results[f'{resolution}_compressed'][hyperparameter]:
                    print(f"FLOPs: {results[f'{resolution}_compressed'][hyperparameter]['flops']/1e9:.2f} GFLOPs")
    
        if verbose:
            print("\n" + "="*50)
            print("PERFORMANCE COMPARISON")
            print("="*50)
            print("\nRelative increase in error (compressed vs original):")
            print("-"*50)
        results["Comparison"][hyperparameter] = {}

        base_results = results[f"{resolution}_base"]
        comp_results = results[f"{resolution}_compressed"]
        l2_change_percentage = (comp_results[hyperparameter]['l2_loss'] / base_results['l2_loss'] - 1) * 100
        results["Comparison"][hyperparameter]["l2_loss_increase"] = l2_change_percentage
        print(f"{resolution}x{resolution} - L2: {l2_change_percentage:.2f}%")
        
        # Performance comparison if tracking enabled
        if track_performance:
            if 'runtime' in base_results and 'runtime' in comp_results[hyperparameter]:
                if comp_results[hyperparameter]['runtime'] == 0:
                    speedup = 0
                else:
                    speedup = (base_results['runtime'] / comp_results[hyperparameter]['runtime']) * 100
                results["Comparison"][hyperparameter]["run_time_speed_up"] = speedup
                print(f"{resolution}x{resolution} - Runtime Speedup: {speedup:.2f}x")
            
            if 'model_size_mb' in base_results and 'model_size_mb' in comp_results[hyperparameter]:
                model_size_reduction = (1 - comp_results[hyperparameter]['model_size_mb'] / base_results['model_size_mb']) * 100
                results["Comparison"][hyperparameter]["model_size_reduction"] = model_size_reduction
                print(f"{resolution}x{resolution} - Model Size Reduction: {model_size_reduction:.2f}%")
            
            if 'peak_memory_mb' in base_results and 'peak_memory_mb' in comp_results[hyperparameter]:
                memory_reduction = (1 - comp_results[hyperparameter]['peak_memory_mb'] / base_results['peak_memory_mb']) * 100
                results["Comparison"][hyperparameter]["peak_memory_reduction"] = memory_reduction
                print(f"{resolution}x{resolution} - Peak Memory Reduction: {memory_reduction:.2f}%")
            
            if 'flops' in base_results and 'flops' in comp_results[hyperparameter]:
                flops_reduction = (1 - comp_results[hyperparameter]['flops'] / base_results['flops']) * 100
                results["Comparison"][hyperparameter]["flops_reduction"] = flops_reduction
                print(f"{resolution}x{resolution} - FLOPs Reduction: {flops_reduction:.2f}%")
            
    
    return results



def compare_models_hyperparams(model1, model2s, hyperparameters, test_loaders, data_processor, device, 
                  model1_name="Original Model", model2_name="Compressed Model",
                  verbose=True, track_performance=False, evaluation_params=None):
    """Compare performance between two models across different resolutions.
    
    Args:
        model1: First model to evaluate (e.g., original model)
        model2: Second model to evaluate (e.g., compressed model)
        test_loaders: Dict of test loaders for different resolutions
        data_processor: Data processor for the dataset
        device: Device to run evaluation on
        model1_name: Name for the first model (default: "Original Model")
        model2_name: Name for the second model (default: "Compressed Model")
        verbose: Whether to print detailed results (default: True)
        track_performance: Whether to track runtime, memory usage and FLOPs (default: False)
    """
    results = {}
    results["Comparison"] = {}
    resolution, loader = next(iter(test_loaders.items()))
    results[f"{resolution}_compressed"] = {}
    if verbose:
        print("\n" + "="*50)
        print(f"{model1_name.upper()} EVALUATION")
        print("="*50)
    
    for resolution, loader in test_loaders.items():
        if verbose:
            print(f"\nResults on {resolution}x{resolution} resolution")
            print("-"*30)
        results[f"{resolution}_base"] = evaluate_model(model1, loader, data_processor, device, 
                                                     track_performance=track_performance, 
                                                     evaluation_params=evaluation_params)
        if verbose:
            print(f"L2 Loss: {results[f'{resolution}_base']['l2_loss']:.6f}")
            if track_performance:
                if 'runtime' in results[f'{resolution}_base']:
                    print(f"Avg Runtime per batch: {results[f'{resolution}_base']['runtime']*1000:.2f} ms")
                if 'model_size_mb' in results[f'{resolution}_base']:
                    print(f"Model Size: {results[f'{resolution}_base']['model_size_mb']:.2f} MB")
                if 'peak_memory_mb' in results[f'{resolution}_base']:
                    print(f"Peak Memory Usage: {results[f'{resolution}_base']['peak_memory_mb']:.2f} MB")
                if 'flops' in results[f'{resolution}_base']:
                    print(f"FLOPs: {results[f'{resolution}_base']['flops']/1e9:.2f} GFLOPs")
    
    for k in range(len(hyperparameters)):
        hyperparameter = hyperparameters[k]
        model2 = model2s[k]
        if verbose:
            print("\n")
            print("<"+"-"*50, f"{model2_name.upper()} EVALUATION with hyperparam({str(hyperparameter)})", 50*"-"+">")
        
        if hasattr(model2, 'get_compression_stats') and verbose:
            stats = model2.get_compression_stats()
            print(f"\nModel sparsity: {stats['sparsity']:.2%}")

            if 'original_size' in stats:
                print(f"Original size: {stats['original_size']} bytes")
            if 'quantized_size' in stats:
                print(f"Quantized size: {stats['quantized_size']} bytes")
            if 'compression_ratio' in stats:
                print(f"Compression ratio: {stats['compression_ratio']:.2f}")
            if 'dyquantized_layers' in stats:
                print("Quantized layers:")
                for layer_name in stats['dyquantized_layers']:
                    print(f" - {layer_name}")
    


        results[f"{resolution}_compressed"] = {}
        if verbose:
            print(f"\nResults on {resolution}x{resolution} resolution")
            print("-"*30)
        results[f"{resolution}_compressed"][hyperparameter] = evaluate_model(model2, loader, data_processor, device,
                                                        track_performance=track_performance,
                                                        evaluation_params=evaluation_params)
        if verbose:
            print(f"L2 Loss: {results[f'{resolution}_compressed'][hyperparameter]['l2_loss']:.6f}")
            if track_performance:
                if 'runtime' in results[f'{resolution}_compressed'][hyperparameter]:
                    print(f"Avg Runtime per batch: {results[f'{resolution}_compressed'][hyperparameter]['runtime']*1000:.2f} ms")
                if 'model_size_mb' in results[f'{resolution}_compressed'][hyperparameter]:
                    print(f"Model Size: {results[f'{resolution}_compressed'][hyperparameter]['model_size_mb']:.2f} MB")
                if 'peak_memory_mb' in results[f'{resolution}_compressed'][hyperparameter]:
                    print(f"Peak Memory Usage: {results[f'{resolution}_compressed'][hyperparameter]['peak_memory_mb']:.2f} MB")
                if 'flops' in results[f'{resolution}_compressed'][hyperparameter]:
                    print(f"FLOPs: {results[f'{resolution}_compressed'][hyperparameter]['flops']/1e9:.2f} GFLOPs")
    
        if verbose:
            print("\n" + "="*50)
            print("PERFORMANCE COMPARISON")
            print("="*50)
            print("\nRelative increase in error (compressed vs original):")
            print("-"*50)
        results["Comparison"][hyperparameter] = {}

        base_results = results[f"{resolution}_base"]
        comp_results = results[f"{resolution}_compressed"]
        l2_change_percentage = (comp_results[hyperparameter]['l2_loss'] / base_results['l2_loss'] - 1) * 100
        results["Comparison"][hyperparameter]["l2_loss_increase"] = l2_change_percentage
        print(f"{resolution}x{resolution} - L2: {l2_change_percentage:.2f}%")
        
        # Performance comparison if tracking enabled
        if track_performance:
            if 'runtime' in base_results and 'runtime' in comp_results[hyperparameter]:
                if comp_results[hyperparameter]['runtime'] == 0:
                    speedup = 0
                else:
                    speedup = (base_results['runtime'] / comp_results[hyperparameter]['runtime']) * 100
                results["Comparison"][hyperparameter]["run_time_speed_up"] = speedup
                print(f"{resolution}x{resolution} - Runtime Speedup: {speedup:.2f}x")
            
            if 'model_size_mb' in base_results and 'model_size_mb' in comp_results[hyperparameter]:
                model_size_reduction = (1 - comp_results[hyperparameter]['model_size_mb'] / base_results['model_size_mb']) * 100
                results["Comparison"][hyperparameter]["model_size_reduction"] = model_size_reduction
                print(f"{resolution}x{resolution} - Model Size Reduction: {model_size_reduction:.2f}%")
            
            if 'peak_memory_mb' in base_results and 'peak_memory_mb' in comp_results[hyperparameter]:
                memory_reduction = (1 - comp_results[hyperparameter]['peak_memory_mb'] / base_results['peak_memory_mb']) * 100
                results["Comparison"][hyperparameter]["peak_memory_reduction"] = memory_reduction
                print(f"{resolution}x{resolution} - Peak Memory Reduction: {memory_reduction:.2f}%")
            
            if 'flops' in base_results and 'flops' in comp_results[hyperparameter]:
                flops_reduction = (1 - comp_results[hyperparameter]['flops'] / base_results['flops']) * 100
                results["Comparison"][hyperparameter]["flops_reduction"] = flops_reduction
                print(f"{resolution}x{resolution} - FLOPs Reduction: {flops_reduction:.2f}%")
            
    return results
