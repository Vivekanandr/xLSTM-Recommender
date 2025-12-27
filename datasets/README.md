
**Experiment Setup and Datasets:-**

<img width="475" height="443" alt="image" src="https://github.com/user-attachments/assets/bad54bc6-6c73-437a-8d21-fe54bd5af376" />


**Datasets:**

1. Amazon Software data usually refers to the large-scale Amazon Product Review datasets (reviews, ratings, timestamps, and product metadata) widely used for recommender system research. They capture user–item interactions across millions of products and enable benchmarking of collaborative and content-based recommendation models.

		https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
		
		Software	reviews (459,436 reviews)	metadata (26,815 products)

2. MUSIK4all is a massive music interaction dataset (≈228M events, 119,140 users, TSV File) containing user–track play counts and timestamps. It is designed for music recommender systems, supporting temporal modeling, user history analysis, and large-scale evaluation.

Before Filter:

		Total rows: 252984396
		Unique users: 119140
		Unique tracks: 56512
		Time span: 1970-01-01 01:00:36 → 2020-03-20 12:59:51
		Avg. interactions per user: 2123.42
		Min / Max interactions per user: 1 / 243384

Data Sampling (Filters): 

		A. Filter users <5 events,
		B. cap max 1000 events per user,
		C. hash-select ~5% of users,
		D. output to Parquet (~10M rows).

After Sampling Filter:

		total_rows:	2533266
		unique_users:	4584
		unique_tracks: 	51291
		min_timestamp:	2/22/2005 22:45
		max_timestamp: 	3/20/2020 12:59
		avg_interactions_per_user: 	552.6
		min_interactions_per_user: 	5
		max_interactions_per_user:	1000

**Data Sampling Techniques:**
				
				A. Random Sampling → Select a fraction p of rows uniformly (e.g., USING SAMPLE 1% in DuckDB); fast but may fragment user timelines.
				B. Stratified Sampling → Sample proportionally within groups (e.g., per user/item category); implemented with groupby + sample() in Pandas/Polars.
				C. Systematic Sampling → Pick every k-th record after a random offset; efficient for ordered files but risky if patterns exist.
				D. Time-based Sampling → Filter interactions by a timestamp window (e.g., WHERE ts >= NOW() - INTERVAL '90 days'); preserves temporal recency.
				E. User-based Hash Sampling → Deterministic subset of users via hash (e.g., hash(user_id) % 100 = 0); keeps complete histories of selected users.
				F. Per-user Last-K Sampling → Take last K events per user using window functions (ROW_NUMBER() OVER (PARTITION BY user ORDER BY ts DESC)); reduces data but preserves recency.
				G. Storage formats → For scale, write sampled subsets to Parquet/ZSTD (columnar, compressed) for fast reloads vs. raw TSV/CSV.
				H. Best practice → Use hash or last-K sampling for recommender research, time-based for evaluation splits, and combine with Parquet for speed.

