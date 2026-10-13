# 📱 Mobile Product Segmentation and Recommendation System

End-to-end data analytics and machine learning pipeline on 50,000 global mobile phone reviews (7 brands, 22 models): data cleaning, EDA, K-Means customer segmentation, a cosine-similarity phone recommender, and a multi-page Streamlit app.

## 🗂️ Project Structure

```
Mobile Product Segmentation and Recommendation System/
│
├── data/
│   ├── raw/                          # Original Global Mobile Reviews CSV
│   ├── cleaned/                      # mobile_reviews_cleaned.csv (0 nulls, typed)
│   └── processed/                    # reviews_encoded, reviews_scaled, reviews_clustered, product_level
│
├── notebooks/
│   ├── initial_analysis.ipynb        # Column exploration, null audit, cleaning, imputation
│   ├── eda.ipynb                     # Brand, price, rating, age and time analysis
│   ├── feature_engineering.ipynb     # avg_spec_score, one-hot encoding, scaling, product table
│   ├── clustering.ipynb              # Elbow + silhouette, K-Means (k=4), segment profiling
│   └── recommender.ipynb             # Cosine similarity recommender + validation
│
├── models/
│   ├── scaler.pkl                    # StandardScaler for the review-level features
│   ├── kmeans_k4.pkl                 # Trained K-Means model
│   └── recommender.pkl               # Scaler, weights and feature list for the recommender
│
├── src/
│   └── recommender.py                # build_similarity() and recommend()
│
├── pages/
│   ├── 1_EDA_Dashboard.py
│   └── 2_Segments_and_Recommender.py
│
├── main.py                           # Streamlit entry point (home page)
├── data_loader.py                    # Cached data and model loaders
├── project_documents.md              # Detailed process and decisions
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Running the Streamlit App

Run from the project root folder:

```bash
pip install -r requirements.txt
streamlit run main.py
```

## 🖥️ How to Use the Dashboard

**EDA Dashboard page:** Use the sidebar filters (brand, country, year) and browse the tabs — distribution, price, ratings (model averages and correlation heatmap), age, and time trends — with KPIs for reviews, average price, average rating and positive %.

**Segments & Recommender page:**
1. **Segments tab:** view segment share, segment profiles, brand mix per segment and a 2D scatter of the clusters. Enter a customer's price, spec score, rating, helpful votes and age, then click **Predict segment**.
2. **Recommender tab:** pick a phone, choose how many recommendations you want, optionally restrict to the same price tier, and view similar phones with their price tier and similarity score.

## 📊 Results

| Task | Method | Result |
|---|---|---|
| Customer segmentation | K-Means (k=4, 5 scaled features) | 4 segments; silhouette ≈ 0.20 |
| Phone recommendation | Cosine similarity (price weighted ×3) | Recommended phones avg $134 from the chosen phone's price vs $290 for random pairs |

**The 4 segments:**

| Segment | Size | Avg price | Avg rating |
|---|---|---|---|
| Budget – Dissatisfied | 15,428 | $506 | 1.97 |
| Mid-range – Young Enthusiasts | 14,680 | $680 | 4.11 |
| Premium – Dissatisfied | 8,789 | $1,083 | 2.27 |
| Mid-range – Mature Satisfied | 11,103 | $647 | 4.01 |

## 🔍 Key Findings

- Price is set by brand: Realme ~$393 → Xiaomi → Motorola → OnePlus → Google → Samsung → Apple ~$1,103.
- Ratings are ~3.1 for every brand, model, country and age group, and price does not affect rating.
- Reviewers skew young (~70% aged 18–34), but price, rating and brand preference are the same across ages.
- Premium buyers pay the most but rate among the lowest — a value-for-money risk for Apple and Samsung.
- Mid-range customers are the most satisfied; the two mid-range segments differ by age and engagement, not brand.

## 📌 Notes

- Review-level data (50,000 rows) is used for clustering; the recommender uses a 22-row model-level table (average price, ratings, positive %).
- The elbow curve has no sharp bend and the silhouette is highest at k=2 (≈0.29); k=4 was chosen as the brief asks for 4 segments and it gives interpretable budget / mid-range / premium groups. The moderate silhouette is expected for such uniform data.
- Encoded brand and country columns are kept out of K-Means (brand already determines price) and used to interpret the segments instead.
- Model-level ratings differ by only ~0.1, so price is weighted ×3 in the recommender to prevent noise from driving the results.
- `price_local`, `exchange_rate_to_usd` and `currency` are excluded from modeling because they mix currencies; `price_usd` is used throughout.
- Model names are title-cased in the cleaned data (e.g. `Iphone 14`); the recommender lookup ignores capitalization.

See `project_documents.md` for the full process and decisions.