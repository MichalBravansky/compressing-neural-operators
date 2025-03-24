import torch
import time
from neuralop.losses import LpLoss, H1Loss
from neuralop.models import FNO
from neuralop.data.datasets import load_darcy_flow_small
from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info


def evaluate_model(model, dataloader, data_processor, device='cuda', track_performance=False, verbose=False):
    """
    Evaluates model performance with optional tjracking of runtime, memory usage, and FLOPs.

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

    Returns
    -------
    dict
        Dictionary containing metrics including 'l2_loss' and performance metrics.
    """
    model.eval()
    total_l2_loss = 0.0
    total_h1_loss = 0.0
    total_runtime = 0.0
    batch_count = 0

    flops_counted = False
    flops = 0

    model_size_mb = 0
    if track_performance and torch.cuda.is_available():
        model = model.to('cpu')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        before_model_load = torch.cuda.memory_allocated(device)

        model = model.to(device)
        model_size_mb = (torch.cuda.memory_allocated(
            device) - before_model_load) / (1024 * 1024)

        torch.cuda.reset_peak_memory_stats(device)
        start_memory = torch.cuda.memory_allocated(device)
    else:
        model = model.to(device)

    if data_processor is not None:
        data_processor = data_processor.to(device)
        data_processor.eval()

    l2_loss = LpLoss(d=2, p=2, reduction='mean')
    h1_loss = H1Loss(d=2, reduction='mean')

    with torch.no_grad():
        for batch in dataloader:
            if data_processor is not None:
                processed_data = data_processor.preprocess(batch)
            else:
                processed_data = {k: v.to(device)
                                  for k, v in batch.items() if torch.is_tensor(v)}

            # Measure FLOPs on first batch only using ptflops
            if track_performance and not flops_counted:
                try:
                    # Create an input constructor that returns the processed data dictionary
                    def input_constructor(input_res):
                        return {k: v.clone() for k, v in processed_data.items() if torch.is_tensor(v)}

                    # Get model complexity using ptflops - note that input_res is unused in our constructor
                    macs, params = get_model_complexity_info(
                        model, (1,), input_constructor=input_constructor,
                        as_strings=False, print_per_layer_stat=False, verbose=verbose,
                        backend='aten'  # 'aten' backend is more comprehensive
                    )

                    # Convert MACs to FLOPs (multiply by 2)
                    flops = macs * 2
                    flops_counted = True

                except Exception as e:
                    if verbose:
                        print(f"Error measuring FLOPs with ptflops: {e}")
                    flops = 0

            # Track runtime
            if track_performance:
                start_time = time.time()

            out = model(**processed_data)

            if track_performance:
                # Wait for CUDA operations to finish
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                end_time = time.time()
                total_runtime += (end_time - start_time)

            if data_processor is not None:
                out, processed_data = data_processor.postprocess(
                    out, processed_data)

            total_l2_loss += l2_loss(out, processed_data['y']).item()
            total_h1_loss += h1_loss(out, processed_data['y']).item()
            batch_count += 1

    avg_l2_loss = total_l2_loss / len(dataloader)
    avg_h1_loss = total_h1_loss / len(dataloader)

    result = {'l2_loss': avg_l2_loss, 'h1_loss': avg_h1_loss}

    if track_performance:
        result['runtime'] = total_runtime / max(batch_count, 1)

        if model_size_mb > 0:
            result['model_size_mb'] = model_size_mb

        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated(
                device) - start_memory
            result['peak_memory_mb'] = peak_memory / (1024 * 1024)

        if flops > 0:
            result['flops'] = flops

    return result


def compare_models(model1, model2, test_loaders, data_processor, device,
                   model1_name="Original Model", model2_name="Compressed Model",
                   verbose=True, track_performance=False):
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
                                                       track_performance=track_performance)
        if verbose:
            print(f"L2 Loss: {results[f'{resolution}_base']['l2_loss']:.6f}")
            if track_performance:
                if 'runtime' in results[f'{resolution}_base']:
                    print(
                        f"Avg Runtime per batch: {results[f'{resolution}_base']['runtime']*1000:.2f} ms")
                if 'model_size_mb' in results[f'{resolution}_base']:
                    print(
                        f"Model Size: {results[f'{resolution}_base']['model_size_mb']:.2f} MB")
                if 'peak_memory_mb' in results[f'{resolution}_base']:
                    print(
                        f"Peak Memory Usage: {results[f'{resolution}_base']['peak_memory_mb']:.2f} MB")
                if 'flops' in results[f'{resolution}_base']:
                    print(
                        f"FLOPs: {results[f'{resolution}_base']['flops']/1e9:.2f} GFLOPs")

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
                                                             track_performance=track_performance)
        if verbose:
            print(
                f"L2 Loss: {results[f'{resolution}_compressed']['l2_loss']:.6f}")
            if track_performance:
                if 'runtime' in results[f'{resolution}_compressed']:
                    print(
                        f"Avg Runtime per batch: {results[f'{resolution}_compressed']['runtime']*1000:.2f} ms")
                if 'model_size_mb' in results[f'{resolution}_compressed']:
                    print(
                        f"Model Size: {results[f'{resolution}_compressed']['model_size_mb']:.2f} MB")
                if 'peak_memory_mb' in results[f'{resolution}_compressed']:
                    print(
                        f"Peak Memory Usage: {results[f'{resolution}_compressed']['peak_memory_mb']:.2f} MB")
                if 'flops' in results[f'{resolution}_compressed']:
                    print(
                        f"FLOPs: {results[f'{resolution}_compressed']['flops']/1e9:.2f} GFLOPs")

    if verbose:
        print("\n" + "="*50)
        print("PERFORMANCE COMPARISON")
        print("="*50)
        print("\nRelative increase in error (compressed vs original):")
        print("-"*50)

        for resolution in test_loaders.keys():
            base_results = results[f"{resolution}_base"]
            comp_results = results[f"{resolution}_compressed"]
            print(
                f"{resolution}x{resolution} - L2: {(comp_results['l2_loss']/base_results['l2_loss'] - 1)*100:.2f}%")

            # Performance comparison if tracking enabled
            if track_performance:
                if 'runtime' in base_results and 'runtime' in comp_results:
                    speedup = base_results['runtime'] / comp_results['runtime']
                    print(
                        f"{resolution}x{resolution} - Runtime Speedup: {speedup:.2f}x")

                if 'model_size_mb' in base_results and 'model_size_mb' in comp_results:
                    model_size_reduction = (
                        1 - comp_results['model_size_mb'] / base_results['model_size_mb']) * 100
                    print(
                        f"{resolution}x{resolution} - Model Size Reduction: {model_size_reduction:.2f}%")

                if 'peak_memory_mb' in base_results and 'peak_memory_mb' in comp_results:
                    memory_reduction = (
                        1 - comp_results['peak_memory_mb'] / base_results['peak_memory_mb']) * 100
                    print(
                        f"{resolution}x{resolution} - Peak Memory Reduction: {memory_reduction:.2f}%")

                if 'flops' in base_results and 'flops' in comp_results:
                    flops_reduction = (
                        1 - comp_results['flops'] / base_results['flops']) * 100
                    print(
                        f"{resolution}x{resolution} - FLOPs Reduction: {flops_reduction:.2f}%")

    return results


def optional_fno(resolution):
    # Low resolution FNO
    if resolution == "low":
        fno_model = FNO(
            in_channels=1,
            out_channels=1,
            n_modes=(16, 16),
            hidden_channels=32,
            projection_channel_ratio=2,
            n_layers=4,
            skip="linear",
            norm="group_norm",
            implementation="factorized",
            separable=False,
            factorization=None,
            rank=1.0,
            domain_padding=None,
            stabilizer=None,
            dropout=0.0)
        fno_model.load_state_dict(torch.load(
            "models/model-fno-darcy-16-resolution-2025-02-05-19-55.pt", weights_only=False))
        fno_model.eval()
        train_loader, test_loaders, data_processor = load_darcy_flow_small(
            n_train=1000,
            batch_size=16,
            test_resolutions=[16, 32],
            n_tests=[100, 50],
            test_batch_sizes=[16, 16],
            encode_input=False,
            encode_output=False,
        )
        return fno_model, train_loader, test_loaders, data_processor

    elif resolution == "high":
        fno_model = FNO(
            in_channels=1,
            out_channels=1,
            n_modes=(32, 32),
            hidden_channels=64,
            projection_channel_ratio=2,
            n_layers=5,
            skip="linear",
            norm="group_norm",
            implementation="factorized",
            separable=False,
            factorization=None,
            rank=1.0,
            domain_padding=None,
            stabilizer=None,
            dropout=0.0)

        fno_model.load_state_dict(torch.load(
            "models/model-fno-darcy-16-resolution-2025-03-04-18-48.pt", weights_only=False))
        fno_model.eval()

        train_loader, test_loaders, data_processor = load_darcy_flow_small(
            n_train=100,
            batch_size=16,
            test_resolutions=[128],
            n_tests=[10000],
            test_batch_sizes=[1],
            encode_input=True,
            encode_output=True,
        )
        return fno_model, train_loader, test_loaders, data_processor
