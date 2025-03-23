===========================================
Model Compression on Neural Operators
===========================================

Introduction
----------------
This repository contains the implementation for our paper on compressing neural operators to enhance their efficiency without compromising performance. Neural operators are designed to learn mappings between function spaces and have shown significant benefits in terms of accuracy and generalization. However, scaling these models introduces considerable deployment challenges, including increased latency and substantial computational resource demands.

Our work presents the first comprehensive study on compressing neural operators, demonstrating that post-training compression is not only feasible but yields substantial efficiency gains.

Methodology
----------------
We systematically investigate multiple compression techniques:

1. **Magnitude-based weight pruning**: Removing less important weights based on their absolute values
2. **Low-rank matrix factorization**: Decomposing weight matrices into lower-rank approximations
3. **Layer pruning**: Removing entire layers while maintaining overall network functionality
4. **Static quantization**: Converting model weights to lower precision formats
5. **Dynamic quantization**: Runtime conversion of activations to lower precision

Through extensive experimentation on various neural operators, including CoDAno and Fourier Neural Operator (FNO), we demonstrate significant efficiency improvements with minimal accuracy loss.

Usage Instructions
------------------
Follow these steps to get started with the codebase:

1. Set the python path (for lab machine, run it each time you reconnect)
   
.. code ::

   setenv PYTHONPATH `pwd`

2. Download dataset
   
.. code ::

   python compression/dataset_download.py
   python compression/wandb_download/dataset_download.py

3. Download models
   
.. code ::

   python compression/wandb_download/weights_download.py

4. Evaluation Example (on CoDAno)
   
.. code :: 

   python compression/evaluation_docano.py

Compression Examples
-------------------

Try different compression techniques:

.. code ::

   # Magnitude-based pruning
   python compression/prune_model.py --model_type fno --sparsity 0.5
   
   # Low-rank factorization
   python compression/factorize_model.py --model_type codano --rank_ratio 0.1
   
   # Quantization
   python compression/quantize_model.py --model_type fno --precision int8

Key Results
-------------
Our experiments demonstrate that:

- Neural operators can be compressed with minimal performance degradation
- Compression benefits become more pronounced with larger model scales
- Different compression techniques offer varying tradeoffs between computation time, memory usage, and accuracy
- Post-training compression provides a flexible approach to optimizing neural operators for resource-constrained environments

The released code and datasets enable further research and facilitate the practical integration of efficient neural operators into computational workflows across diverse domains and deployment scenarios.