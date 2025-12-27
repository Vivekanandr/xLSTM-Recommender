
**Model Architecture Hyperparameters at a glance:** (xLSTMLargeConfig — Advanced Configuration with Theoretical Justifications)

1. embedding_dim=128
	• What it does: Specifies the dimensionality of learned vector representations for discrete input tokens (e.g., movie IDs).
	• Why: A higher embedding dimension increases representational capacity, enabling the model to capture more latent semantic features. The embedding layer projects sparse one-hot input vectors into a continuous, dense space where semantic similarity correlates with vector proximity.
	• Common alternatives:
		○ 64: Lower capacity, faster convergence.
		○ 256/512: Useful in large item vocabularies to prevent underfitting.
	• When to use: Scale with dataset complexity. Use 128–256 for medium-size datasets with rich item metadata.
	Recommended: MovieLens 100K → 64 or 128, MovieLens 1M → 128 or 256, MovieLens 20M → 256 or 512
	

2. num_heads=2
	• What it does: Defines the number of parallel attention heads in multi-head attention modules within the xLSTM blocks.
	• Why: Multi-head attention decomposes the representation space into subspaces, allowing the model to attend to information from multiple perspectives simultaneously. This increases its ability to capture heterogeneous temporal dependencies.
	• Common alternatives:
		○ 1: Deactivates multi-head decomposition, reducing model complexity.
		○ 4/8: Enables modeling finer-grained patterns across modalities or positional contexts.
	• When to use: Increase when sequences are long or contain multiple intertwined dependencies (e.g., genre + recency + popularity).
	Recommended: MovieLens 100K → 1 or 2, MovieLens 1M → 2 or 4, MovieLens 20M → 4 or 8
	

3. num_blocks=2
	• What it does: Sets the number of stacked xLSTM layers.
	• Why: Deeper architectures allow hierarchical learning where lower layers capture local dependencies and higher layers model abstract, long-range patterns. This improves generalization and capacity to capture complex sequence dynamics.
	• Common alternatives:
		○ 1: Suitable for shallow tasks or small data regimes.
		○ 3+: Improves abstraction, suitable for deep sequence modeling like session-based or hierarchical recommendation.
	• When to use: Start with 2. Increase depth if the model underfits or fails to capture long-term user behavior trends.
	Recommended: MovieLens 100K → 1 or 2, MovieLens 1M → 2 or 3,  MovieLens 20M → 3 or 4
	

4. vocab_size=num_items + 1
	• What it does: Defines the size of the input vocabulary (items) including padding.
	• Why: Essential for allocating the correct size of embedding and output matrices. The +1 accounts for a sentinel token (e.g., <PAD>), critical for batching variable-length sequences.
	• When to use: Always match to dataset; padding index typically uses ID 0.
	

5. return_last_states=True
	• What it does: Returns only the final hidden state from each sequence.
	• Why: In next-item prediction, only the final timestep matters — intermediate states are irrelevant. Returning only the last state reduces memory and computational overhead during inference.
	• Alternative:
		○ False: Needed for token-level tasks or for attention over the whole sequence in downstream layers.
	• When to use: True for sequence-to-one settings; False for sequence-to-sequence or explainability requirements.
	Recommended: All MovieLens versions → True
	

6. mode="inference"
	• What it does: Controls the internal operation flags — disables dropout, gradient tracking, etc.
	• Why: Reduces unnecessary stochasticity and overhead during evaluation. Ensures deterministic behavior, important for reproducibility and deployment.
	• Alternative:
		○ "training": Activates regularization components like dropout.
	• When to use: Set to "inference" during evaluation or production deployment.
	Recommended: Use "training" when fitting the model. Use "inference" during evaluation or deployment.
	
7. chunkwise_kernel="chunkwise--triton_xl_chunk"
	• What it does: Specifies the backend kernel used for chunk-based sequence processing in the xLSTM.
	• Why: Chunking enables parallelism over sub-segments of the sequence, reducing latency and memory usage while preserving local context. Triton provides a high-performance, GPU-optimized kernel for this.
	• Alternatives:
		○ "chunkwise--native": CPU-friendly but slower and less parallelized.
	• When to use: Always prefer Triton if targeting GPU execution and high throughput.
	• All MovieLens versions → "chunkwise--triton_xl_chunk"
	

8. sequence_kernel="native_sequence__triton"
	• What it does: Determines the kernel used for processing full sequences end-to-end.
	• Why: In sequence modeling, kernel efficiency dictates overall throughput. Triton kernels can fuse operations and minimize memory transfers on GPUs.
	• Alternatives:
		○ "native_sequence__torch": More debuggable but less performant.
	• When to use: Triton for production/research; Torch for debugging and CPU contexts.
	• All MovieLens versions → "native_sequence__triton"
	

9. step_kernel="triton"
	• What it does: Kernel for token-by-token (autoregressive) prediction.
	• Why: In online inference, the model predicts one step at a time. Efficient step kernels minimize latency and memory reuse overhead.
	• Alternatives:
		○ "torch": Simplified fallback, better suited for testing and interpretability.
	• When to use: Triton in real-time systems or batch decoding tasks.

