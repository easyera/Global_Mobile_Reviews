import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import load_cleaned

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")
st.title("📊 EDA Dashboard")

df = load_cleaned()
df['year'] = df['review_date'].dt.year
df['age_group'] = pd.cut(df['age'], bins=[17, 24, 34, 44, 54, 65],
                         labels=['18-24', '25-34', '35-44', '45-54', '55-65'])

# ---- sidebar filters ----
st.sidebar.header("Filters")
brands = st.sidebar.multiselect("Brand", sorted(df['brand'].unique()), default=sorted(df['brand'].unique()))
countries = st.sidebar.multiselect("Country", sorted(df['country'].unique()), default=sorted(df['country'].unique()))
years = st.sidebar.slider("Year", int(df['year'].min()), int(df['year'].max()),
                          (int(df['year'].min()), int(df['year'].max())))

f = df[df['brand'].isin(brands) & df['country'].isin(countries) & df['year'].between(*years)]

if f.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ---- KPIs ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews", f"{len(f):,}")
c2.metric("Avg price (USD)", f"${f['price_usd'].mean():,.0f}")
c3.metric("Avg rating", f"{f['rating'].mean():.2f}")
c4.metric("Positive %", f"{(f['sentiment'] == 'Positive').mean() * 100:.1f}%")

# ---- tabs ----
t1, t2, t3, t4, t5 = st.tabs(["Distribution", "Price", "Ratings", "Age", "Time"])

with t1:
    a, b = st.columns(2)
    a.plotly_chart(px.bar(f['brand'].value_counts().reset_index(), x='brand', y='count', title='Reviews by brand'), use_container_width=True)
    b.plotly_chart(px.pie(f, names='sentiment', title='Sentiment share'), use_container_width=True)
    a.plotly_chart(px.bar(f['country'].value_counts().reset_index(), x='country', y='count', title='Reviews by country'), use_container_width=True)
    b.plotly_chart(px.bar(f['source'].value_counts().reset_index(), x='source', y='count', title='Reviews by source'), use_container_width=True)

with t2:
    st.plotly_chart(px.histogram(f, x='price_usd', nbins=40, title='Price distribution (USD)'), use_container_width=True)
    order = f.groupby('brand')['price_usd'].median().sort_values().index
    st.plotly_chart(px.box(f, x='brand', y='price_usd', category_orders={'brand': list(order)}, title='Price by brand'), use_container_width=True)

with t3:
    m = (f.groupby(['brand', 'model']).agg(avg_price=('price_usd', 'mean'), avg_rating=('rating', 'mean'),
                                          reviews=('rating', 'size')).round(2).reset_index())
    st.plotly_chart(px.bar(m.sort_values('avg_price'), x='avg_price', y='model', color='brand', orientation='h', title='Average price by model'), use_container_width=True)
    st.plotly_chart(px.bar(m.sort_values('avg_rating'), x='avg_rating', y='model', color='brand', orientation='h', title='Average rating by model', range_x=[2.8, 3.4]), use_container_width=True)
    st.plotly_chart(px.imshow(f[['age', 'price_usd', 'rating', 'battery_life_rating', 'camera_rating', 'performance_rating',
                                 'design_rating', 'display_rating', 'helpful_votes']].corr().round(2),
                              text_auto=True, color_continuous_scale='RdBu_r', title='Correlation heatmap'), use_container_width=True)

with t4:
    st.plotly_chart(px.histogram(f, x='age', nbins=24, title='Age distribution'), use_container_width=True)
    g = f.groupby('age_group', observed=True).agg(reviews=('rating', 'size'), avg_price=('price_usd', 'mean'), avg_rating=('rating', 'mean')).round(2)
    st.dataframe(g, use_container_width=True)
    share = pd.crosstab(f['age_group'], f['brand'], normalize='index').reset_index().melt(id_vars='age_group')
    st.plotly_chart(px.bar(share, x='age_group', y='value', color='brand', title='Brand share by age group'), use_container_width=True)

with t5:
    mo = f.set_index('review_date').resample('MS').agg(reviews=('rating', 'size'), avg_rating=('rating', 'mean')).reset_index()
    st.plotly_chart(px.line(mo, x='review_date', y='reviews', markers=True, title='Reviews per month'), use_container_width=True)
    st.plotly_chart(px.line(mo, x='review_date', y='avg_rating', markers=True, title='Average rating per month'), use_container_width=True)