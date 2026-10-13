# Mobile Product Segmentation and Recommendation System — Project Documentation

This document explains the process followed and the key decisions made at each stage of the project, including why certain approaches were chosen over alternatives.

---

## 1. Business Understanding

**What the data is:** 50,000 customer reviews of smartphones from the Global Mobile Reviews dataset — 7 brands (Apple, Samsung, Google, OnePlus, Motorola, Xiaomi, Realme), 22 models, reviewers from multiple countries, reviews dated Oct 2022 – Oct 2025.

**Business questions this project answers:**
- What market segments exist (budget, mid-range, premium) and what do customers in each segment look like?
- How do price, ratings, age and brand relate to each other?
- Which phones are similar to a phone a customer already likes?

**Key column distinctions clarified early:**
- `price_usd` is the price in one common currency; `price_local` is the same price in the reviewer's local currency (`price_local = price_usd × exchange_rate_to_usd`). They carry the same information.
- `rating` is the overall rating (1–5); `battery`, `camera`, `performance`, `design`, `display` are the five sub-ratings (1–5), used as the product "specifications" because the dataset has no hardware spec columns.
- `helpful_votes` is used as the engagement score.
- The data is **review-level** (one row = one review), not product-level. Two tables are therefore needed: a review-level table for customer segmentation and a model-level table (one row per phone) for the recommender.

---

## 2. Issue Identification

Before cleaning, the data was audited using a standard framework: structure → missing values → duplicates → categorical/string consistency → numeric distribution → domain/logic rules.

**Findings:**
- 22 columns, 50,000 rows. `review_id` and `customer_name` are identifiers with no analytical value.
- Nulls in five columns: `price_usd` (2,450), `price_local` (2,431), `rating` (2,453), `sentiment` (2,445), `source` (2,448).
- ~22% of rows (11,058) had at least one null — too many to drop; column-by-column imputation was needed.
- No duplicate rows.
- `price_local` was stored as text with currency symbols, thousand separators and trailing spaces (`$1,010.68 `, `₹27996.73`, `د.إ2425.65`). The AED symbol itself contains a dot, so a simple "keep digits and dots" cleanup would have broken ~5,900 values.
- `review_date` was a string; sub-ratings and counts were stored as 64-bit integers.
- `price_local` and `exchange_rate_to_usd` mix many currencies, so their overall min/max/mean are not meaningful.

---

## 3. Data Cleaning — Decisions & Reasoning

| Column(s) | Decision | Why |
|---|---|---|
| `review_id`, `customer_name` | Dropped | Identifiers; no analytical value |
| `price_local` | Parsed to float by removing thousand separators and extracting only the number at the end of the string | Handles `$1,010.68 `, `€256.43` and the AED symbol that contains a dot |
| `price_usd` (2,450 nulls) | Step 1: filled with `price_local ÷ exchange_rate_to_usd` (2,327 rows). Step 2: remaining 123 rows (both prices missing) filled with the median price of the same model | The two price columns are exactly linked (tested on ~39,000 complete rows: difference < 0.01), so the first step is a calculation, not a guess. Median is used for the rest because prices vary widely within a model (e.g. Galaxy S24: $401–$1,399) |
| `price_local` (remaining nulls) | Rebuilt as `price_usd × exchange_rate_to_usd` | Keeps both price columns complete and consistent |
| `rating` (2,453 nulls) | Filled with the rounded mean of the 5 sub-ratings, clipped to 1–5 | Each sub-rating correlates ~0.76 with `rating`; a row-level estimate is better than a group average |
| `sentiment` (2,445 nulls) | Filled with the most common sentiment for that row's `rating` (1→Negative, 2→Neutral, 3/4/5→Positive) | Sentiment follows rating closely; done after `rating` was complete, so the order of imputation matters |
| `source` (2,448 nulls) | Filled with `Unknown` | Sources are spread evenly (~9.5k each) with no link to any other column — nothing to predict it from, so no value is invented |
| `review_date` | Converted to datetime (`%m/%d/%Y`) | All 50,000 values parse correctly |
| Numeric columns | `age` → int16; sub-ratings and `helpful_votes` → int8; `verified_purchase` → bool | Correct types, smaller memory |
| String columns | Kept as `str`, whitespace and casing standardized (e.g. `Ebay`, `Oneplus`, `Iphone 14`) | Consistent categories; note the casing when looking up model names |
| Column order | Date and flag first, then qualitative columns, then quantitative columns | Easier to read and analyse |
| Outlier check | No rows removed | `price_usd` $180–$1,500 (no IQR outliers), `age` 18–65, ratings 1–5, `helpful_votes` 0–17 — all valid |

Result: **0 nulls, 0 duplicates, 50,000 rows**. No rows were dropped.

---

## 4. Exploratory Data Analysis — Key Insights

1. **The dataset is balanced** — brand, country and review source are evenly distributed, so comparisons between them are fair. Sentiment is Positive 56%, Neutral 25%, Negative 19%.
2. **Price is set by brand, not by model.** Average price: Realme ~$393, Xiaomi ~$450, Motorola ~$507, OnePlus ~$672, Google ~$805, Samsung ~$901, Apple ~$1,103. Models within one brand cost about the same, and `price_usd` has no outliers ($180–$1,500).
3. **Ratings are almost identical across products** — every model averages ~3.1 overall (camera ~2.7). Top-rated and low-rated models differ by only about 0.1, so there is no clear best or worst model.
4. **Price does not affect rating** — the price-by-rating boxplots overlap at every rating level; correlation between price and rating is ~0.
5. **Sub-ratings move together** — each of the five sub-ratings correlates ~0.76 with the overall rating.
6. **No time trend** — review volume and average rating are flat from 2022 to 2025.
7. **Reviewers skew young but behave the same at every age** — ~70% are aged 18–34 (18–24: 15,509; 25–34: 19,168; 35–44: 11,980; 45–54: 3,072; 55–65: 271). Average price (~$690), rating (~3.1) and brand share are the same in every age group.

These insights directly shaped feature selection: price and the satisfaction measures (sub-ratings, rating, helpful votes) are the signals that separate customers; age, time, country and source add little.

---

## 5. Feature Engineering — Decisions & Reasoning

- **`avg_spec_score`** = mean of the five sub-ratings per review — one spec-quality score per review.
- **`verified_purchase`** converted to 0/1.
- **Categorical encoding:** `brand`, `country`, `sentiment` and `source` were one-hot encoded (dataset after encoding: 50,000 × 41). `currency` was left out because it carries the same information as `country`; `model` is only used as the row label in the product table (22 values).
- **K-Means features (5):** `price_usd`, `avg_spec_score`, `rating`, `helpful_votes`, `age`.
  - The five sub-ratings were replaced by `avg_spec_score` — they are correlated (~0.76), and including all five would count the same signal five times.
  - The encoded brand/country columns were deliberately **kept out of K-Means**: brand already determines price, so it would be counted twice. They are used afterwards to profile and interpret the clusters.
  - `price_local`, `exchange_rate_to_usd` and `currency` were excluded because they mix currencies.
- **Scaling:** `StandardScaler` (mean 0, std 1). Without it, price (hundreds) would dominate ratings (1–5) in distance calculations. The fitted scaler is saved for the app.
- **Product table (22 rows × 12 columns):** one row per model with review count, average price, average rating, the five sub-ratings (battery, camera, performance, design, display), average helpful votes and positive-sentiment %. Used for the recommender.

---

## 6. Modeling — Customer Segmentation (K-Means)

**Approach:** K-Means on the 5 scaled features, k tested from 2 to 8 using the elbow method (inertia) and the silhouette score (10,000-row sample).

**Result:** The elbow curve falls smoothly with no sharp bend, and the silhouette score is highest at k=2 (≈0.29) and flattens from k=4 onward (≈0.20). Overall separation is moderate, which is expected for data this uniform.

**Decision:** k=4 was chosen. k=2 only splits "low vs high" (too coarse for budget / mid-range / premium), the project brief asks for 4 segments, and the silhouette is stable from k=4 upward. The moderate silhouette is documented rather than hidden.

**Segments found:**

| Segment | Size | Avg price | Avg spec score | Avg rating | Avg votes | Avg age | Brand mix |
|---|---|---|---|---|---|---|---|
| Budget – Dissatisfied | 15,428 | $506 | 1.72 | 1.97 | 2.09 | 29 | Mostly Xiaomi, Realme, Motorola |
| Mid-range – Young Enthusiasts | 14,680 | $680 | 3.61 | 4.11 | 5.32 | 24 | Evenly mixed |
| Premium – Dissatisfied | 8,789 | $1,083 | 1.96 | 2.27 | 2.44 | 30 | 40% Apple, 27% Samsung |
| Mid-range – Mature Satisfied | 11,103 | $647 | 3.52 | 4.01 | 4.55 | 39 | Evenly mixed |

**Insights:**
- Price tier and satisfaction drive the segments: mid-range customers are the most satisfied, while both budget and premium customers are dissatisfied.
- The premium segment pays the most (~$1,083) but rates among the lowest — a value-for-money risk for Apple and Samsung.
- The two mid-range segments are separated by age and engagement, not by brand (their brand mix is nearly even).

---

## 7. Modeling — Recommender (Cosine Similarity)

**Approach:** Each of the 22 models is a row of 8 features (average price, average rating, the five sub-ratings, positive %), scaled with `StandardScaler`. Cosine similarity between every pair of models gives a 22 × 22 similarity table; the top-N most similar models are recommended. `helpful_votes` was dropped from this table because it measures engagement, not a product feature.

**Key decision — price weighting:** Model-level ratings differ by only ~0.1 (std 0.02–0.04), while price has a std of ~$250. Plain scaling would make those tiny rating differences count as much as price, so recommendations would be driven by noise. Price was multiplied by a weight of 3, so it counts about as much as all the other features combined; ratings act as tie-breakers.

**Price tier (integration with segmentation):** Each model is labelled Budget (< $550), Mid-range ($550–$1,000) or Premium (> $1,000), matching the price levels of the K-Means segments. The tier is shown next to every recommendation, and the app can restrict results to the same tier.

**Example:** For Iphone 14, the top results are Iphone 13 (Premium, similarity 0.897), Galaxy Note 20, Pixel 7A, Galaxy Z Flip and Pixel 8 (all Mid-range).

**Validation:** For each of the 22 models, the average price gap to its 5 recommendations was **$133.5**, versus **$290.4** between random pairs of models — the recommended phones are about 54% closer in price than chance.

---

## 8. End-to-End Application (Streamlit)

Built as a multi-page app:
- **Home (`main.py`)** — project summary and headline numbers (reviews, brands, models, countries).
- **EDA Dashboard** — sidebar filters (brand, country, year); KPIs; tabs for distribution, price, ratings (model averages and correlation heatmap), age and time trends. Charts are built with Plotly.
- **Segments & Recommender** — one page for both models:
  - *Segments:* segment share, profile table, brand mix by segment, a 2D PCA scatter, and a form that predicts the segment for a custom customer (price, spec score, rating, votes, age) using the saved scaler and K-Means model.
  - *Recommender:* pick a phone, choose how many recommendations, optionally restrict to the same price tier, and see a table plus a similarity chart.

**Key implementation decisions:**
- The scaler, K-Means model and recommender settings (scaler, weights, feature list) are saved as `.pkl` files, so the app applies exactly the same transformations as the notebooks.
- Recommendation logic lives in `src/recommender.py`, defined once and imported by the app.
- The app is run from the project root so relative paths resolve.

---

## 9. Summary of Key Decisions

| Decision | Reasoning |
|---|---|
| Fill `price_usd` from `price_local ÷ exchange_rate` first | The relationship is exact, so this is a calculation rather than an estimate |
| Impute in order: price → rating → sentiment | Sentiment is predicted from the (completed) rating |
| Rating filled from the mean of the sub-ratings | Row-level, strongly correlated (~0.76); better than a group average |
| `source` filled with `Unknown` | Evenly spread with no pattern — predicting it would invent data |
| Drop `price_local`, `exchange_rate_to_usd`, `currency` from modeling | Mixed currencies; same information as `price_usd` and `country` |
| One `avg_spec_score` instead of five sub-ratings | Avoids counting one correlated signal five times |
| Keep encoded brand/country out of K-Means | Brand already determines price (double counting); used to interpret clusters instead |
| Cluster at review level, recommend at model level | 22 model-level rows are too few and too similar for meaningful clusters; 50,000 reviews give real variation |
| k=4 despite a moderate silhouette | Required by the brief, gives interpretable budget / mid-range / premium segments, silhouette stable from k=4 up |
| Weight price ×3 in the recommender | Model ratings are almost identical; without weighting, noise would drive the results |
| Validate the recommender against random pairs | Shows recommendations are ~54% closer in price than chance |

---

## 10. Limitations

- The data is very uniform: ratings, sentiment, time and age patterns look nearly identical across brands and models, so the main real signal is price (set by brand). Segments are therefore driven mostly by price level and satisfaction.
- Silhouette scores are moderate (≈0.20 at k=4); segments overlap.
- The recommender is mostly price-driven because model-level ratings barely differ.
- Imputed sentiment values (2,445 rows) follow the most common sentiment for each rating, which slightly reinforces the existing rating–sentiment link.
- The dataset has no hardware specification columns; the five sub-ratings stand in as product "specs".