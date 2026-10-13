import pandas as pd
import joblib
import streamlit as st

@st.cache_data
def load_cleaned():
    return pd.read_csv('data/cleaned/mobile_reviews_cleaned.csv', parse_dates=['review_date'])

@st.cache_data
def load_clustered():
    return pd.read_csv('data/processed/reviews_clustered.csv')

@st.cache_data
def load_products():
    return pd.read_csv('data/processed/product_level.csv')

@st.cache_resource
def load_models():
    return {
        'scaler': joblib.load('models/scaler.pkl'),
        'kmeans': joblib.load('models/kmeans_k4.pkl'),
        'rec': joblib.load('models/recommender.pkl'),
    }