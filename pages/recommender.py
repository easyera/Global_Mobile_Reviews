import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from data_loader import load_clustered, load_products, load_models
from src.recommender import build_similarity, recommend

st.set_page_config(page_title="Segments & Recommender", page_icon="🎯", layout="wide")
st.title("🎯 Customer Segments & Phone Recommender")

cl = load_clustered()
prod = load_products()
models = load_models()
feats = ['price_usd', 'avg_spec_score', 'rating', 'helpful_votes', 'age']
seg_map = dict(cl[['cluster', 'segment']].drop_duplicates().values)

tab1, tab2 = st.tabs(["Segments (K-Means)", "Recommender (Cosine similarity)"])

# ================= SEGMENTS =================
with tab1:
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.pie(cl, names='segment', title='Segment share'), use_container_width=True)

    prof = cl.groupby('segment')[feats].mean().round(2)
    prof.insert(0, 'size', cl['segment'].value_counts())
    c2.subheader("Segment profile")
    c2.dataframe(prof, use_container_width=True)

    brand_cols = [c for c in cl.columns if c.startswith('brand_')]
    mix = cl.groupby('segment')[brand_cols].mean().round(2)
    mix.columns = [c.replace('brand_', '') for c in mix.columns]
    st.plotly_chart(px.imshow(mix, text_auto=True, color_continuous_scale='Blues', title='Brand mix by segment'),
                    use_container_width=True)

    @st.cache_data
    def pca_points():
        d = cl.sample(6000, random_state=42)
        pts = PCA(n_components=2, random_state=42).fit_transform(models['scaler'].transform(d[feats]))
        out = pd.DataFrame(pts, columns=['PC1', 'PC2'])
        out['segment'] = d['segment'].values
        return out

    st.plotly_chart(px.scatter(pca_points(), x='PC1', y='PC2', color='segment', opacity=0.6,
                               title='Segments in 2D (PCA)'), use_container_width=True)

    st.subheader("Which segment is this customer in?")
    a, b, c, d, e = st.columns(5)
    price = a.number_input("Price (USD)", 150, 1600, 600)
    spec = b.slider("Avg spec score", 1.0, 5.0, 3.0, 0.1)
    rating = c.slider("Rating", 1, 5, 3)
    votes = d.slider("Helpful votes", 0, 20, 3)
    age = e.slider("Age", 18, 65, 30)

    if st.button("Predict segment"):
        x = pd.DataFrame([[price, spec, rating, votes, age]], columns=feats)
        label = models['kmeans'].predict(models['scaler'].transform(x))[0]
        st.success(f"Segment: {seg_map[label]}")

# ================= RECOMMENDER =================
with tab2:
    sim_df = build_similarity(prod, models['rec'])

    a, b, c = st.columns([2, 1, 1])
    choice = a.selectbox("Pick a phone", sorted(prod['model']))
    n = b.slider("How many recommendations", 3, 10, 5)
    same = c.checkbox("Same price tier only")

    info = prod.set_index('model').loc[choice]
    st.write(f"**{choice}** ({info['brand']}) | {info['price_tier']} | ${info['avg_price']:,.0f} | rating {info['avg_rating']:.2f}")

    rec = recommend(prod, sim_df, choice, n, same)
    st.dataframe(rec, use_container_width=True)
    st.plotly_chart(px.bar(rec, x='similarity', y='model', color='price_tier', orientation='h',
                           title='Similarity score'), use_container_width=True)