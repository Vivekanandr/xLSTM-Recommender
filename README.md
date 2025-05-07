# Recommenders:

At its core, a **recommendation engine** uses computer algorithms to predict and suggest items of interest to users based on their past behaviors and contextual data. On the deep learning front, xLSTM (extended LSTM) and Transformers have been evolved with latest architectures in recent years and plays a very important role in Large Language Models...

Since recommenders have became an essential part of our daily digital experiences, in this paper we will be leveraging XLSTM architecture based recommenders and will be comparing the results with other modern architectures like Autotransformers, Recurring Neural Networks (RNN) and other Matrix Factorization Methods. xLSTM incorporates architectural enhancements like attention mechanism, gating improvments and bidirectional capabilities. It will be impactful due to several unique aspects when to compared to the tradional successful methods. 

**Existing Approaches:**
**Transformers Based:** Uses the self-attention mechanism from transformer architectures (like in GPT or BERT) to model the user-item interactions. It also captures the complex sequential patterns in user behavior (e.g., purchase history or clicks) and finally makes personalized recommendations by understanding the contextual relationships between items and users. Unlike, RNN which process information sequencially (one step at a time), transformer process information in parallel utilizing the self attentio mechanism which results in faster computation precerving long term dependencies.

A transformer-based recommender system uses an embedding layer to convert the users and items into dense vector representations. It applies self-attention layers to capture complex dependencies and sequential relationships in user-item interactions, enhanced by positional encodings to preserve and keep the order of actions. The final output layer computes scores or rankings to predict the most relevant items for each user.

**Matrix Factorization Based:** It decompose a user-item interaction matrix into two smaller matrices: one representing users and the other representing items. These matrices capture latent factors (hidden patterns) that explain the user preferences and item characteristics. By reconstructing the original matrix, the system predicts how much a user might like an unseen item. Some techniques include SIngular Value Decomposition (SVD), Non-Negative Matrix Factorization (NMF) and Probabilistic Matrix Factorization (PMF).

**Proposed Hybrid Methods & Noval Approaches:** xLSTM (extended LSTM) incorporates architectural enhancements like attention mechanism, gating improvments and bidirectional capabilities. It will be impactful due to several unique aspects when to compared to the traditional successful methods. 

Below are the two major types in Recommenders:
1. Collaborative Filtering (User to User), and
2. Content Based Filtering (Product to Product).

Different Simple methods to identify user similarities:
1. Correlations, 2. Cosine Similarities, 3. Jaccard Similarities, 4. Euclidean Distance, 5. Hamming Distance, 6. Manhatten Distance, 7. Bhattachryya Distance, 8. Neural Network Embeddings (Collaborative Filtering), 9. Kullback Leibler divergence, 10. Embeddings and Latent Features, 11. Sequence-Based Similarity, 12. Deep Collaborative Filtering with Embeddings (via Neural Networks), 13. Transformer Models for Sequential Recommendations (e.g. BERT4Rec), and 14. Other Hybrid Approches.

**Data Sources:** Here, we will be leveraging RecBole libraries to explore various models and to develop more customizable one. 

List of Different Varieties of Datasets: (Ref: https://recbole.io/)

![image](https://github.com/user-attachments/assets/e842adf0-6eaa-48b7-9ffa-68312db0788e)

-------------------------------------------------------------------------------------------------------------------------------------
**Dataset 1: MovieLENS (100-K)**

**Model: Bert4Rec**

**Input Datasets:**
![image](https://github.com/user-attachments/assets/9a728e92-2d62-4f6e-b2cd-96080a482eb1)

**Performance Optimization:** Comparision and Performance Results For All Approaches:
1st set of results (To be fine tuned further)
![image](https://github.com/user-attachments/assets/c73f24ed-737f-40c0-b530-436b40d56b75)


**xLSTM Model Flow Chart:**

Model Explnation in 23 steps: 

Step 1: User watches a sequence of movies: e.g., \[Die Hard, Terminator, Lethal Weapon].

* Why: To learn temporal preferences by modeling user behavior over a time-ordered sequence of interactions. This reflects the dynamic evolution of user interests in a sequential recommendation system.
* How: Real-world logs from MovieLens dataset are parsed per user and timestamp to reconstruct watch histories.

Step 2: Each movie title is mapped to an internal index using item\_to\_idx, becoming e.g., \[12, 45, 7].

* Why: Deep learning models require fixed-size numerical inputs; categorical values must be encoded as integers for downstream embedding.
* How: A bijective mapping (dictionary) translates movie names to internal numeric IDs for efficient indexing and lookup.

Step 3: This index sequence is truncated or adjusted to fit a maximum input length (e.g., last 50 movies).

* Why: Neural models have finite memory and processing budgets. Truncation ensures computational feasibility and uniform input size.
* How: Sequences longer than 50 are sliced to retain only the most recent items, assuming recent behaviors are more indicative.

Step 4: The sequence is zero-padded at the start to maintain consistent length: e.g., \[0, 0, ..., 12, 45, 7].

* Why: Padded sequences ensure all inputs in a batch are the same length, enabling vectorized computation.
* How: Padding tokens (index 0) are added to the beginning of shorter sequences to reach the max length.

Step 5: The padded sequence is converted into a PyTorch tensor of shape (1, 50).

* Why: Tensors are required to interface with PyTorch-based models; they are GPU-compatible data containers.
* How: Python lists are wrapped with `torch.tensor()` to create the appropriate dimensional structure for model input.

Step 6: This tensor is passed to an embedding layer to convert indices to vectors of dim 128, shape becomes (1, 50, 128).

* Why: Embeddings transform discrete items into continuous vector spaces where semantic similarity can be learned.
* How: Each item ID is used as an index into a learnable weight matrix, returning its corresponding vector representation.

Step 7: These embeddings capture semantic information about each movie.

* Why: Capturing latent factors like genre, popularity, or user affinity improves generalization.
* How: The embedding layer learns these representations during training via gradient descent.

Step 8: The embedded tensor is passed through the first xLSTM block, preserving full sequence output: shape (1, 50, 128).

* Why: Temporal models like xLSTM retain ordering and context over time, critical for modeling user sequences.
* How: The xLSTM processes each timestep sequentially but in parallelizable chunks, returning contextualized outputs.

Step 9: This xLSTM block models temporal context and complex sequential patterns in movie viewing behavior. Unlike traditional LSTM, xLSTM introduces chunkwise attention and block-wise memory updates for better parallelism and long-range dependency tracking. It leverages high-performance kernels (e.g., Triton) for scalability and speed. xLSTM is designed to work well in autoregressive and inference modes with minimal memory bottlenecks.

* Why: xLSTM enhances efficiency and accuracy by capturing deeper temporal dependencies and enabling GPU-optimized computation.
* How: Chunked processing reduces recurrent bottlenecks, while memory routing ensures long-term dependencies are preserved.

Step 10: Output is passed to a second xLSTM block that returns only the last hidden state: shape (1, 128).

* Why: The final state condenses all prior contextual information into a fixed-size latent representation.
* How: Only the output at the last timestep (position 50) is extracted for prediction.

Step 11: This hidden state is a compressed representation of the user's full watch history.

* Why: It forms a holistic latent profile summarizing long- and short-term interests.
* How: The hidden vector is treated as a feature encoding of the entire sequence for final prediction.

Step 12: The output is fed into a dense (fully connected) layer that outputs raw logits: shape (1, vocab\_size).

* Why: Dense layers enable transformation from latent user space to the full item probability space.
* How: A weight matrix projects the 128-dim vector into the number of available items (e.g., 951 movies).

Step 13: These logits are scores for each possible movie in the dataset.

* Why: Logits serve as pre-softmax signals reflecting raw model confidence before normalization.
* How: Each score indicates how strongly the model believes an item is the next in sequence.

Step 14: A softmax layer converts logits into probabilities summing to 1.

* Why: Probabilistic interpretation is essential for ranking and evaluation metrics.
* How: Softmax uses the exponential of logits to derive a categorical distribution over all movies.

Step 15: The output probabilities indicate the model's confidence for each movie being the next.

* Why: Ranking is done based on relative probabilities to recommend top-k candidates.
* How: A probability vector is created with each index representing likelihood of that movie.

Step 16: The top-k probabilities are selected (e.g., top-10), and their indices are sorted in descending order.

* Why: Reduces computational complexity by focusing on high-probability items.
* How: `torch.topk()` or similar function selects highest probability indices.

Step 17: The top index (e.g., 202) is considered the most likely next movie.

* Why: It represents the model's argmax prediction — the single most confident output.
* How: Index with highest softmax value is selected and marked for recommendation.

Step 18: This index is mapped back to the original movie title using idx\_to\_item (e.g., 202 -> Speed).

* Why: Predictions need to be human-readable for deployment in user interfaces.
* How: Reverse mapping dictionary is applied to convert index to title.

Step 19: The model recommends this top movie (Speed) as the next likely movie the user will watch.

* Why: Providing accurate next-item recommendations increases engagement and satisfaction.
* How: Top prediction is surfaced in application dashboards or personalized lists.

Step 20: The MovieLens 100K dataset is first sorted by user and timestamp. Each user's sequence is split into:

* Training: all but last 2 movies,
* Validation: sequences predicting the second-last movie,
* Test: sequence predicting the last movie.
* Why: Sequential splitting mirrors online prediction tasks, ensuring no future leakage.
* How: Sequences are chronologically segmented into task-specific sets based on user ID and timestamp.

Step 21: The training objective uses CrossEntropyLoss between predicted logits and the actual next movie index.

* Why: Cross-entropy is optimal for classification tasks and penalizes deviations from the true label.
* How: The true movie index is compared to the softmax output and gradients are backpropagated accordingly.

Step 22: During evaluation, the model predicts probabilities across all movies. Recall\@10, MRR\@10, and NDCG\@10 are calculated by comparing the ranked predictions with the true next movie.

* Why: These metrics capture ranking quality and relevance, essential for recommendation systems.
* How: For each sample, the true movie's rank in the predicted top-k list is measured and aggregated.

Step 23: This pipeline can be repeated for other users, continuously learning patterns across movie sequences.

* Why: Model retraining or online learning allows adapting to evolving user preferences.
* How: New interaction logs are appended to training data and the model is updated accordingly.


![image](https://github.com/user-attachments/assets/1e9163a7-ca51-4c08-ab80-8cb229713983)


**xLSTM Results: (MovieLENS 1M)**

![image](https://github.com/user-attachments/assets/a98049b0-17d2-45b4-88e0-841777c7ea71)

**Results Interpretations:**

Recall@10 = 0.2932, Meaning: In ~29.3% of cases, the true next item appears in the top 10 predictions. Solid result for a baseline xLSTM model. It means if we recommend 10 movies, ~3 times out of 10, the correct one will be in that list.

MRR@10 = 0.1266, Mean Reciprocal Rank: Measures the position of the first correct item in the top-10 list. On average, the correct item appears around position 8 (1/0.1266 ≈ 7.9). Higher MRR means better ranking of correct items.

NDCG@10 = 0.1655, Normalized Discounted Cumulative Gain: Measures both relevance and position in the ranked list. A higher value means the model is not just retrieving the right item, but ranking it closer to the top. At ~16.5% of the optimal ranking gain. Good enough for an initial run,better results when its finetunes over 200 epochs.


xLSTM Results: (MovieLENS 100-k)

![image](https://github.com/user-attachments/assets/e02f8dd3-2619-42c8-a375-c9fa423ea5c7)

![image](https://github.com/user-attachments/assets/0ae5172a-30a1-45c7-8c28-bb75eca255af)

Other Model: 
![image](https://github.com/user-attachments/assets/b2f78535-fe0f-419d-822d-57626102bd5f)

**Evaluation Metrics:** To evaluate the model accuracy Recall 5, 10, Precision, NDCG will be used mainly. 

Recall = How many relevant items recommended/Total No. of relevant items **available** (measures the relevance. )

Precision: How many relevant items recommended//Total No. of items **recommended** (measures the accuracy.)

**Normalized Discounted Combined Gain (NDGC):** For Ranking.

**Epochs:** How many times we process our complete data until we reach final/optimum goal. 

**Learning rate:**, How fast did we adjust our weights to reach that optimum level.

**Cold Start Problem** (For new users when we don't have data, 192 users): 
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

![image](https://github.com/user-attachments/assets/d157ae60-54c2-41e8-9bf3-e79e1250bc1b)

Few Output Results:
![image](https://github.com/user-attachments/assets/ae41618e-a913-44c1-80d6-76d15a3faba3)


**Logit Score:** Direct Score, before applying any activation funtions, non bounded ( can be larger and can go larger negative values). Higher the logit score, better the prediction is.

**Probability:** Derived from logit score after applying softmax function (always between 0 to 1), probability is calculated across all the Items in the list, so it might seem to be less, distributed across all of them. 


-------------------------------------------------------------------------------------------------------------------------------------
Model 2: (GRU4Rec - MovieLENS 100K)
![image](https://github.com/user-attachments/assets/5efd7700-083e-4dfc-a73d-576a17de20a4)


--------------------------------------------------------------------------------------------------------------------------------------
Model 3: (SAS4Rec - MovieLENS 100K)

![image](https://github.com/user-attachments/assets/12d95ab8-3fc4-400c-a16f-9fdcfceef5d8)

--------------------------------------------------------------------------------------------------------------------------------------

Model 4: (Bert4Rec Steam Data)

![image](https://github.com/user-attachments/assets/e3fd6c5f-c5a5-4e20-b1cc-c78a91a442f2)

---------------------------------------------------------------------------------------

**Baseline Model Results (For comparision)**

![image](https://github.com/user-attachments/assets/2106998a-f865-4218-a5e0-f1510e75907e)

------------------------------------------------------------------------------------------------

**Recbole - Major Classifications:**

Four Classifications: 

1. General Recommendation (GR): Netflix use case which we discussed above. The interaction of users and items is the only data that can be used by model. Trained on implicit feedback data and evaluated using top-n recommendation. Collaborative filter (CF) based models are classified here. 

2. Content-aware Recommendation: Amazon use case. Click-through rate prediction, CTR prediction. The dataset is explicit and contains label field. Evaluation conducted by binary classification.

3. Sequential Recommendation: Spotify, similar to time series problem, which we discussed earlier. The task of SR (next-item recommendation) is the same as GR which sorts a list of items according to preference. History interactions are organized in sequences and the model tends to characterize the sequential data. Session-based recommendation are also included here.

4. Knowledge-based Recommendation: Knowledge-based recommendation introduces an external knowledge graph to enhance general or sequential recommendation.

SEO (Search Engine Optimization) and SEM techniques may also be merged, along with Google Adsense and adwords, to improve user experience further. 

**Scope** For Recommendation Engines In Various Sectors:
1. Energy Sectors, (Energy Saving Programs, Substations, CO2 Emission, Solar, Grid Automation, Sensor Meters, Electrical Products and HVAC transmission)
2. Banking and Fintech sectors, (Wealth Management, Customized Credit Products, Investment Portfolio's, Equities, and Insurance plan recommendations)
3. Technology and Service sectors, (Ecommerce Products)
4. Entertainment and Gamification, (Custom Localization and Immersive Experience)
5. Food, Beverages & Agriculture Industry (AI-based recommendations for improved crop yield, agricultural products, custom fertilizers, supply and demand forecasting, as well as weather and climate change insights using satellite data.)
6. Healthcare and Pharmaceutical, 
7. Aviation and Transportation, and 
8. Other Specialized Sectors. 

**Few Hugging Face models to be tested:**
1. Transformers4Rec by NVIDIA: Integrates with Hugging Face Transformers, enabling the application of transformer architectures to sequential and session-based recommendation tasks.

2. RecGPT: RecGPT is a domain-adapted large language model specifically trained for text-based recommendation tasks.

**MLops:** ML Flow, Wandb for model training, deployment and testing

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



----------------------------------------

**Rough Workouts:**

Singular Value Decomposition: (SVD to decompose the original matrices into three smaller matrices)
![image](https://github.com/user-attachments/assets/50bac5cb-eae8-44f2-8ca0-270a69233eef)
*Slight variations in the input matrices noted.


Take Away from SVD:

1. Higher positive values in the reconstructed matrix indicate stronger recommendations.

2. For users or hobbies not strongly represented, values will remain closer to 0 or negative.


Business Context: 

Value Proposition: The core concept and unique benefit is to **recommend the best similar product/service to the end user** and to help them. 

It is also estimated that the **market scope** for recommender system is expected to be approx 20-28 billion by 2030, currently in 2024 valued approx 6 billion US dollars. 

Audio Podcast Version: https://www.dropbox.com/scl/fi/zv511ysp0ecdaqbo9nskp/Recommender-Systems_-Architectures-Applications-and-Market-Analysis.wav?rlkey=3u9za3bbogvc0506ubxohxe2w&st=gy3ekapc&dl=0
