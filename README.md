# Recommender Systems

At its core, a **recommendation engine** uses computer algorithms to predict and suggest items of interest to users based on their past behaviors and contextual data. On the deep learning front, xLSTM (extended LSTM) and Transformers have been evolved with latest architectures in recent years and plays a very important role in Large Language Models. 

Since recommenders have became an essential part of our daily digital experiences, in this paper we will be leveraging XLSTM architecture based recommenders and will be comparing the results with other modern architectures like Autotransformers, Recurring Neural Networks (RNN) and other Matrix Factorization Methods. xLSTM incorporates architectural enhancements like attention mechanism, gating improvments and bidirectional capabilities. It will be impactful due to several unique aspects when to compared to the tradional successful methods. 

**Existing Approaches:**
**Transformers Based:** Uses the self-attention mechanism from transformer architectures (like in GPT or BERT) to model the user-item interactions. It also captures the complex sequential patterns in user behavior (e.g., purchase history or clicks) and finally makes personalized recommendations by understanding the contextual relationships between items and users. Unlike, RNN which process information sequencially (one step at a time), transformer process information in parallel utilizing the self attentio mechanism which results in faster computation precerving long term dependencies.

A transformer-based recommender system uses an embedding layer to convert the users and items into dense vector representations. It applies self-attention layers to capture complex dependencies and sequential relationships in user-item interactions, enhanced by positional encodings to preserve and keep the order of actions. The final output layer computes scores or rankings to predict the most relevant items for each user.

**Matrix Factorization Based:** It decompose a user-item interaction matrix into two smaller matrices: one representing users and the other representing items. These matrices capture latent factors (hidden patterns) that explain the user preferences and item characteristics. By reconstructing the original matrix, the system predicts how much a user might like an unseen item. Some techniques include SIngular Value Decomposition (SVD), Non-Negative Matrix Factorization (NMF) and Probabilistic Matrix Factorization (PMF).


**Proposed Hybrid Methods & Noval Approaches:** xLSTM (extended LSTM) incorporates architectural enhancements like attention mechanism, gating improvments and bidirectional capabilities. It will be impactful due to several unique aspects when to compared to the traditional successful methods. 


**Evaluation Metrics:** To evaluate the model accuracy Recall 5, 10, Precision, NDCG will be used mainly. 

Recall = How many relevant items recommended/Total No. of relevant items **available**

It measures the relevance. 

Precision: How many relevant items recommended//Total No. of items **recommended**

It measures the accuracy.

**Normalized Discounted Combined Gain (NDGC):** For Ranking.


**Few Hugging Face models to be tested:**
1. Transformers4Rec by NVIDIA: Integrates with Hugging Face Transformers, enabling the application of transformer architectures to sequential and session-based recommendation tasks.
                  transformer_config = tfr.TransformerBlock(
   
                   d_model=64,  # Embedding dimension
   
                   n_head=4,    # Number of attention heads
   
                   num_layers=2  # Number of transformer layers)

           Embedding dimension, Attention Heads, Transformer layers - Working examples to be added

3. RecGPT: RecGPT is a domain-adapted large language model specifically trained for text-based recommendation tasks.

**References:**
1. https://recbole.io/docs/user_guide/model/general/bpr.html
2. https://recbole.io/
3. https://grouplens.org/datasets/movielens/100k/
4. https://huggingface.co/docs/hub/en/models-the-hub
5. https://www.ibm.com/think/topics/recommendation-engine#:~:text=The%20market%20for%20recommendation%20systems,to%20triple%20in%205%20years.&text=Uncover%20the%20benefits%20of%20AI%20platforms%20that%20enable%20foundation%20model%20customization.
6. https://openai.com/ 
7. https://www.amazon.science/the-history-of-amazons-recommendation-algorithm
8. https://www.amazon.science/code-and-datasets/simrec-mitigating-the-cold-start-problem-in-sequential-recommendation-by-integrating-item-similarity
9. https://github.com/amazon-science
10. xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via Scalar Memories: https://doi.org/10.48550/arXiv.2410.16928
11. xLSTM: Extended Long Short-Term Memory: https://arxiv.org/pdf/2405.04517
12. xLSTM Time : Long-term Time Series Forecasting With xLSTM: https://doi.org/10.48550/arXiv.2407.10240
13. Quaternion Transformer4Rec: Quaternion numbers-based Transformer for recommendation: https://github.com/vanzytay/QuaternionTransformers
14. Recommender Systems: A Primer: https://doi.org/10.48550/arXiv.2302.02579
15. Exploring the Impact of Large Language Models on Recommender Systems: An Extensive Review: https://arxiv.org/pdf/2402.18590
16. Recommender Systems with Generative Retrieval: https://openreview.net/pdf?id=BJ0fQUU32w
17. Attention Is All You Need: https://arxiv.org/abs/1706.03762
