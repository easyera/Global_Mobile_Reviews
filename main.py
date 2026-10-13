import streamlit as st
from data_loader import load_cleaned, load_products

st.set_page_config(page_title="Mobile Segmentation & Recommendation", page_icon="📱", layout="wide")

df = load_cleaned()
prod = load_products()

st.title("📱 Mobile Product Segmentation & Recommendation System")
st.write("An end-to-end analytics project on global mobile phone reviews: EDA, K-Means customer segmentation, and a similarity-based phone recommender.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews", f"{len(df):,}")
c2.metric("Brands", df['brand'].nunique())
c3.metric("Models", df['model'].nunique())
c4.metric("Countries", df['country'].nunique())

st.subheader("Pages")
st.markdown("""
- **EDA Dashboard**: brands, prices, ratings, age and time trends, with filters
- **Segments & Recommender**: the 4 customer segments (K-Means) and the phone recommender (cosine similarity)
""")

st.subheader("Pipeline")
st.markdown("Data cleaning → EDA → Feature engineering → K-Means (k=4) → Cosine similarity recommender → Streamlit app")