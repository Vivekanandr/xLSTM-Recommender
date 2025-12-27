
**Research Topic:** xLSTM Architecture's For Recommendations 

**Research questions:**

This report aims to answer the following four research questions:

RQ1: How does xLSTM’s performance scale with dataset size compared to established architectures like BERT4Rec and SAS4Rec ?.

RQ2: How do sequence length and embedding size influence model performance across different item-popularity levels, and do larger sequences or embeddings improve a model’s ability to make accurate long-tail (less popular) recommendations ?.

RQ3: What trade-offs exist between recommendation accuracy and computational cost as sequence length and model complexity increase?.

RQ4: Embedding Saturation and Utilization: How do different model architectures make effective use of their embedding representations, and does embedding dimensionality lead to better spatial distribution, representation diversity, or improved predictive performance ?.

The primary objective is to evaluate the effectiveness of the xLSTM model across multiple datasets and benchmark it against state-of-the-art baselines using established ranking metrics.

-------------------------------------------------

This script trains a sequential recommender system on a **user-specified/customizable MovieLens dataset** (100K, 1M, 10M, or 20M).

It preprocesses the data, maps user/item IDs, and splits interactions into train/validation/test sequences.

Users can select among **four models: standard LSTM, xLSTM, BERT4REC, SAS4REC** variant with configurable parameters.

The selected model is trained using PyTorch with evaluation metrics like Recall@10, MRR, Hit Rate and NDCG.

After training, the best model is used to predict and display top-10 movie recommendations based on user history.


**Training script that integrates:**

A. Dynamic dataset selection (100K, 1M, 10M, 20M)

B. Multiple model choices (LSTM, xLSTM, BERT4Rec, SASRec)

C. Dataset-specific hyperparameters (xlstm_params, dataloader_params)

D. TensorBoard logging

E. GPU monitoring

F. Evaluation metrics (Recall@10, MRR, NDCG)

G. Early stopping + best model saving

H. Easy-readable prediction logging with movie titles


**Initial Setup Requirements**

A. Install Necessary Packages (in quite mode)

B. Triton Activation For GPU Acceleration (To make sure Triton and GPU Accerleration, to speed up the training process)

C. Select the necessary model and datasets (Model)

D. Run all would work, to change the model and datasets, please adjust the variable in the main script.

E. Script is mainly desinged for Colab Environment, for A100 GPU. Single click solution.

**Methodology**

**Experimental Setup::**
1. Datasets: MovieLens (100K, 1M, 10M, Steam)

2. Models Evaluated: xLSTM, BERT4Rec, SASRec

3. Custom configs for each dataset/model

4. Configuration: Custom hyperparameters tuned for each dataset-model combination

**Systems Features:**

1. GPU-accelerated training (NVIDIA A100, Triton-backed kernels for xLSTM), Comprehensive TensorBoard Logging.

2. Early stopping (patience = 3 epochs) with best model checkpointing

3. Real-time Top-K recommendation outputs with movie title

**Training Workflow:**
1. User and Item ID remapping for compact indexing
2. Temporal sequence splitting (Train/Validation/Test) 
3. Random seeds applied (42, 123, 2023) to ensure statistical reproducibility 
4. Early stopping triggered based on Recall@10 Improvments


**Requirements:**
mlstm_kernels: 2.0.0
xlstm: 2.0.4
torch: 2.7.1
torchvision: 0.22.1
torchaudio: 2.7.1

**Folder: Best Models** ( Contains best model for inferencing, v4 is latest)

**Folder: Runs** (Contains all the recent Run History with 8 different performance attributes (Recall, Hit Rate, GPU Performance, Epoch Run Time, Total Parameters etc.)

**Model Results:**

<img width="757" height="638" alt="image" src="https://github.com/user-attachments/assets/dda0beef-6442-48b4-992d-85706b8df85f" />

<img width="1122" height="735" alt="image" src="https://github.com/user-attachments/assets/bb25793d-1293-4d08-9523-19188bf420b9" />


**Overall Research Findings:**

RQ1 — Performance Scaling Across Dataset Sizes:  xLSTM demonstrates a clear positive scaling trend. While performance on the smallest dataset (ML 100K) drags the Transformer models, xLSTM significantly improves as the interaction histories emerge. On MovieLens 10M, xLSTM reaches Recall@10 values around 31.8 percent, converging closely with BERT4Rec, indicating that its gating mechanisms and the enhanced memory structures leverage medium scale datasets effectively. 

RQ2 — Effects of Sequence Length and Embedding Size: Experiments across sequence lengths of 32, 64, and 128 show that xLSTM exhibits increasing performance variance as sequences grow longer, reflecting its sensitivity to temporal window size. Unlike Transformer baselines, which often compress older interactions into dominant embedding directions, xLSTM maintains stronger temporal fidelity in long sequences due to its recurrent gating structure. This results in improved handling of long-term dependencies and enhanced differentiation of long tail items. Larger embedding dimensions further strengthen this effect on the large datasets, while offering limited benefit in sparse or in the short history domains (Table 5.1).

RQ3 — Accuracy vs. Computational Efficiency Trade-offs: xLSTM introduces a measured trade-off between accuracy and computational cost. Training times are typically 1.5×–2× longer than other baselines, and inference speed is moderate—faster than deep Transformer architectures yet slower than lightweight recurrent models. However, xLSTM avoids the quadratic attention bottleneck of Transformers, offering more predictable scaling in long sequences and large catalog sizes. Overall, xLSTM provides balanced accuracy efficiency characteristics across the datasets (Table 5.1).

RQ4 — Embedding Utilization, Saturation, and Representational Diversity: Embedding geometry analyses reveal that xLSTM makes substantially more effective use of embedding space than Transformer baselines. While BERT4Rec and SAS4Rec exhibits the anisotropic embedding structures driven by the popularity bias, xLSTM produces nearly isotropic embeddings with lower hubness, higher intrinsic dimensionality, and more uniform variance distribution. CKA similarity studies further show that the xLSTM learns fundamentally different, sequence-oriented embedding structures rather than compressing items along global similarity axes.

Overall, xLSTM demonstrates strong scaling behavior (RQ1), clear sensitivity to sequence length and embedding size (RQ2), meaningful efficiency‑accuracy trade-offs (RQ3), and superior embedding utilization compared to Transformer (RQ4). 

**Comprehensive Embedding Geometry Analysis for Sequential Recommenders:**

		A. Embedding Spectrum Analysis 
		B. Variance Distribution & Intrinsic Dimension Study 
		C. Hubness and Popularity Bias Evaluation 
		D. t-SNE Embedding Space Visualization 
		E. Cross-Model Representation Geometry Comparison 
		F. Anisotropy and Isotropy Assessment 
		G. Neighborhood Structure Stability Analysis 
		H. Item Similarity Manifold Exploration


**L2 Norm:-**

Layer Normalization standardizes activations to zero mean and unit variance across feature dimensions, followed by learned scaling and shifting. While it does not impose a strict value range, it strongly constrains activation magnitude, typically resulting in values concentrated within a small interval. **Models such as BERT4Rec apply LayerNorm extensively, leading to tightly controlled embedding norms**. In contrast, **xLSTM applies normalization less frequently** and relies on gated recurrent dynamics, allowing embedding magnitude to play a more prominent representational role. These architectural differences explain the observed variation in embedding L2 norms.

<img width="558" height="383" alt="image" src="https://github.com/user-attachments/assets/f25c8aef-2122-4243-b019-6628808c470d" />


“Although the three models have comparable parameter counts, their learned embedding magnitudes differ substantially due to architectural design. BERT4Rec employs extensive Layer Normalization and attention-based mixing, resulting in low-magnitude, direction-focused embeddings. SASRec exhibits moderate embedding norms due to reduced normalization. In contrast, xLSTM relies on gated recurrent memory mechanisms, where embedding magnitude plays a critical role in preserving information over time, leading to higher L2 norms. These differences reflect architectural inductive biases rather than parameter scale.”



<img width="915" height="881" alt="image" src="https://github.com/user-attachments/assets/5e64c75b-f2a4-4370-9787-98ed4caab4ab" />


**Row 1 – Eigenvalue Decay (“spectrum”)**

The strength of each principal component in the embedding covariance.

Interpretation:

		1. BERT4Rec / SASRec decay very steeply → a few dominant directions → anisotropic space (information compressed in few axes).
		2. xLSTM’s curve is much flatter → variance spread across many dimensions → higher intrinsic dimension and better coverage of the vector space.
		3. Flat tail means embeddings retain more independent features.
		4. In Transformers, sharp decay often correlates with popularity or frequency bias.
		5. xLSTM therefore encodes items more uniformly and with richer latent diversity.

**Row 2 – Cumulative Explained Variance**

How many components are needed to explain total variance.

Interpretation:
		
		1. BERT4Rec and SASRec reach ≈ 90 % variance by ~50 dims → heavy redundancy.
		2. xLSTM needs ~200 dims for the same → more distributed information.
		3. A gentle slope indicates broader feature usage and less rank collapse.
		4. This confirms the intrinsic-dimension metrics (≈ 180 / 204 / 250).
		5. In summary, xLSTM = highest representational capacity, BERT4Rec/SASRec = more compact, redundant embeddings.

**Row 3 – Hubness Histograms (k = 10)**

How many times each item appears in other items’ top-10 nearest neighbors.

Interpretation:
		
		1. BERT4Rec / SASRec distributions are extremely right-skewed — a few movies appear hundreds of times ⇒ hub items dominate similarity space.
		2. xLSTM histogram is almost symmetric and much narrower — most items appear roughly equally often.
		3. Lower hubness (Gini ≈ 0.18) ⇒ better fairness and long-tail coverage.
		4. Transformer embeddings likely overfit to popular items.
		5. xLSTM yields a flatter similarity graph, enhancing diversity and mitigating popularity bias.


**Row 4 – t-SNE Projections**

A 2-D nonlinear projection of the 256-D embeddings (cosine distances).

Interpretation:
		
		1. BERT4Rec and SASRec form dense, elliptical blobs — embeddings crowd near a center → again anisotropy and hub formation.
		2. xLSTM plot is more evenly filled, points occupy a ring-like or diffuse shape → isotropy and balanced similarity.
		3. Fewer tight clusters means less genre-specific collapse; features are smoothly spread.
		4. Visually, xLSTM’s space is broader and more uniform.
		5. This geometry supports more stable neighbor retrieval across item types.

-----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------


Parameters:

![image](https://github.com/user-attachments/assets/e04d9637-be3b-4f91-a2f0-478a7f6dae8f)

---------------------------------------

**Recommender System Pipeline:-**

<img width="1151" height="787" alt="image" src="https://github.com/user-attachments/assets/25ce533a-b811-41fe-b4c1-79c314e9d995" />

----------------------------------------

-------------------------------------------------------------------------------------------------------------------------------------


**Training Objective + Optimizer + Scheduler Breakdown at a glance**

Step 1: criterion = nn.CrossEntropyLoss()
	• What it does: Defines the loss function used to measure how well the model’s predictions match the ground truth.
	• Why: CrossEntropyLoss is mathematically equivalent to maximizing the log-likelihood of the true class (movie index) in a multi-class classification setting. It's standard for categorical prediction tasks where only one true label exists.
	• How it works:
		○ Applies log(softmax(logits)) internally.
		○ Penalizes the model if the predicted probability for the true label is low.
	• Alternatives:
		○ nn.NLLLoss: Use with explicit log_softmax output.
		○ FocalLoss: For class-imbalance-sensitive training.
	• Recommended for MovieLens:
		○ MovieLens 100K → CrossEntropyLoss (default, reliable).
		○ MovieLens 1M / 20M → Still effective. Consider FocalLoss if popularity imbalance is extreme.

Step 2: optimizer = optim.Adam(model.parameters(), lr=0.001)
	• What it does: Specifies the optimizer that updates model weights based on computed gradients.
	• Why: Adam (Adaptive Moment Estimation) uses first- and second-order moments to adjust the learning rate per parameter. It converges faster and more stably than SGD in many cases.
	• How it works:
		○ Tracks moving averages of gradients and squared gradients.
		○ Adapts learning rate per parameter dynamically.
	• Alternatives:
		○ SGD: Simpler, requires more tuning.
		○ AdamW: Weight-decay decoupled Adam, more robust for regularization.
		○ RMSProp: Useful in recurrent networks, though less common now.
	• Recommended:
		○ MovieLens 100K → Adam(lr=1e-3)
		○ MovieLens 1M → AdamW(lr=3e-4)
		○ MovieLens 20M → AdamW(lr=1e-4) or scheduled warm-up

Step 3: scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
	• What it does: Decays the learning rate every 5 epochs by multiplying it by 0.5.
	• Why: Learning rate scheduling helps escape local minima early and encourages fine-tuning as training progresses. Reducing LR gradually allows stable convergence.
	• How it works:
		○ Epochs 1–5: LR = 0.001
		○ Epochs 6–10: LR = 0.0005
		○ ... continues halving every step_size
	• Alternatives:
		○ CosineAnnealingLR: Smoothly decays LR to a minimum.
		○ ReduceLROnPlateau: Adaptive decay based on validation loss.
		○ OneCycleLR: Aggressive LR scheduling, good for fast convergence.
	• Recommended:
		○ MovieLens 100K → StepLR(step_size=5, gamma=0.5) 
		○ MovieLens 1M → ReduceLROnPlateau(patience=3) or CosineAnnealing
		○ MovieLens 20M → OneCycleLR for faster training with controlled generalization

Step 4: recall_list, mrr_list, ndcg_list = [], [], []
	• What it does: Initializes lists to store evaluation metrics per epoch for validation and test sets.
	• Why: Tracking Recall@K, MRR@K, and NDCG@K helps monitor ranking quality and ensure model performance is improving.
	• How it works:
		○ After each epoch, predictions are collected.
		○ Top-k metrics are computed and stored.
	• Alternatives:
		○ Store in a dict or log with wandb, TensorBoard, etc.


**Evaluation Metrics:** To evaluate the model accuracy Recall 5, 10, Precision, NDCG will be used mainly. 

Recall = How many relevant items recommended/Total No. of relevant items **available** (measures the relevance. )

Precision: How many relevant items recommended//Total No. of items **recommended** (measures the accuracy.)

**Normalized Discounted Combined Gain (NDGC):** For Ranking.

**Epochs:** How many times we process our complete data until we reach final/optimum goal. 

**Learning rate:**, How fast did we adjust our weights to reach that optimum level.

**Cold Start Problem**: 
Some of the commonly used approaches were:
1. Clustering Approach,
2. Profile Based (Meta Data) Approach,
3. Hierarchical approach, and
4. Novalty or Randomness Approach

**Feature Focus:**

A. User-level features (e.g., user age, gender, occupation)
B. Item-level features (e.g., genres, title metadata)
C. Time or position embeddings beyond fixed positional indices
D. Temporal dynamics (e.g., timestamp-based recency)

**Performance Optimization:**

A. Implement Leave-One-Out Splitting, B. Integrate Negative Sampling.

**Logit Score:** Direct Score, before applying any activation funtions, non bounded ( can be larger and can go larger negative values). Higher the logit score, better the prediction is.

**Probability:** Derived from logit score after applying softmax function (always between 0 to 1), probability is calculated across all the Items in the list, so it might seem to be less, distributed across all of them. 

---------------------------------------------------------------------------------------

**GPU Scaling:-**

We benchmarked GPU inference-time scaling of sequential recommender architectures—BERT4Rec (bidirectional Transformer), SASRec (causal Transformer), and xLSTM (chunkwise recurrent model)—under identical embedding dimension (256), depth (4 blocks), vocabulary, and next-item prediction heads. Models were run in evaluation mode with inference-only forward passes, measuring per-batch latency and throughput as a function of sequence length L at fixed batch size B=32. Sequence lengths were increased up to L=1536, aligned to xLSTM’s 64-token chunk constraint, with GPU synchronization to ensure accurate timing. Transformers exhibit increasing activation and attention costs with L, while xLSTM amortizes recurrence via chunkwise parallel kernels, yielding near-linear memory growth. Observed latency curves were fit on log–log axes to estimate an effective scaling exponent α, capturing empirical runtime growth. BERT4Rec shows α≈0.98, indicating near-linear scaling in this regime due to efficient GPU attention kernels at moderate L. SASRec exhibits α≈1.26, reflecting superlinear growth from causal masking and less efficient attention execution. xLSTM achieves α≈0.64, demonstrating sublinear effective scaling dominated by fixed kernel overhead at small L and efficient chunkwise recurrence at large L. Although xLSTM has higher constant latency at short sequences, its flatter growth enables convergence toward Transformer latency at long L. Overall, results empirically confirm the quadratic sensitivity of attention-based models to sequence length and the long-context efficiency advantage of chunked recurrent architectures during inference.

We have evaluated the inference-time scaling using lightweight proxy implementations of BERT4Rec, SASRec, and xLSTM on GPU, all configured with 256-dimensional embeddings, 4 blocks, and a shared vocabulary of 10,678 items. Pretrained .pt checkpoints were used to initialize item embeddings (and full weights for xLSTM), while the benchmarked architectures and forward passes were defined explicitly in the script. Inference was performed with GPU synchronization to obtain accurate latency measurements. Sequence lengths were swept from 64 to 1536 (aligned to xLSTM’s 64-token chunking constraint) at a fixed batch size of 32. Latency, throughput, and log–log scaling exponents were computed to characterize how inference cost grows with sequence length.

<img width="712" height="545" alt="image" src="https://github.com/user-attachments/assets/88a6af5c-001e-4b32-bd0d-3ff8f7f42285" />

In our GPU inference benchmark at batch size 32, xLSTM shows higher latency than Transformers for short sequences but much flatter growth as sequence length increases. Its effective scaling exponent (α≈0.64) indicates sublinear runtime growth, reflecting amortized chunkwise recurrence. As a result, xLSTM narrows the latency gap at long sequences (L≈1024–1536), where attention-based models degrade more rapidly.

-------------------------------------------

**General Classification of Recommender Systems:**

Sequential Recommendation (SR):- SR focuses on next-item prediction by modeling the temporal ordering of user interactions. These models utilize sequential data to capture evolving user preferences. RNN-based and transformer-based models are generally included in this category, and this is the primary research focus in this thesis work.

General Recommendation (GR):-  These models rely solely on user–item interaction data, typically in the form of implicit feedback. Implicit feedback includes signals that indirectly indicate user preferences, such as clicks, add-to-cart events, purchases, time spent, or interaction frequency.

Content-Aware Recommendation:- These models incorporate additional side information, such as user or item features. They are often applied in click-through rate (CTR) prediction tasks, using explicit feedback and binary classification evaluation. As feature-based methods, they often go beyond raw user–item interactions by including information about users, items, or context.

Knowledge-Based Recommendation:- Utilizes external knowledge graphs to add semantic or structural context beyond interactions.

**MLops:** ML Flow, Wandb for model training, deployment and testing

-------------------------------------------

**References:**

[1] xLSTM: Extended Long Short-Term Memory: https://arxiv.org/pdf/2405.04517

[2] xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via Scalar Memories: https://doi.org/10.48550/arXiv.2410.16928

[3] Amazon Science: https://github.com/amazon-science

[4] xLSTM Time : Long-term Time Series Forecasting With xLSTM: https://doi.org/10.48550/arXiv.2407.10240

[5] Quaternion Transformer4Rec: Quaternion numbers-based Transformer for recommendation: https://github.com/vanzytay/QuaternionTransformers

[6] Recommender Systems: A Primer: https://doi.org/10.48550/arXiv.2302.02579

[7] Exploring the Impact of Large Language Models on Recommender Systems: An Extensive Review: https://arxiv.org/pdf/2402.18590

[8] Recommender Systems with Generative Retrieval: https://openreview.net/pdf?id=BJ0fQUU32w

[9] Attention Is All You Need: https://arxiv.org/abs/1706.03762

[10] Recbole: https://recbole.io

[11] Group Lens: https://grouplens.org/datasets/movielens/100k/

[12] OpenAI: https://openai.com/

[13] Hugging Face: https://huggingface.co/docs/hub/en/models-the-hub

[14] Kreutz, C.K., Schenkel, R. Scientific paper recommendation systems: a literature review of recent publications. Int J Digit Libr 23, 335–369 (2022). https://doi.org/10.1007/s00799-022-00339-w

[15] Recommendation Systems: Algorithms, Challenges, Metrics, and Business Opportunities https://doi.org/10.3390/app10217748

[16] Roy, D., Dutta, M. A systematic review and research perspective on recommender systems. J Big Data 9, 59 (2022). https://doi.org/10.1186/s40537-022-00592-5

[17] A Comprehensive Review of Recommender Systems: Transitioning from Theory to Practice https://doi.org/10.48550/arXiv.2407.13699

[18] Lost in Sequence: Do Large Language Models Understand Sequential Recommendation?: https://arxiv.org/pdf/2502.13909


