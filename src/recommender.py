import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def build_similarity(prod, art):
    feats = art['feat_cols']
    P = pd.DataFrame(art['scaler'].transform(prod[feats]), columns=feats)
    P = P * pd.Series(art['weights'])
    sim = cosine_similarity(P)
    return pd.DataFrame(sim, index=prod['model'], columns=prod['model'])

def recommend(prod, sim_df, model_name, n=5, same_tier=False):
    scores = sim_df[model_name].drop(model_name)
    if same_tier:
        tier = prod.set_index('model').loc[model_name, 'price_tier']
        scores = scores[scores.index.isin(prod.loc[prod['price_tier'] == tier, 'model'])]
    scores = scores.sort_values(ascending=False).head(n)
    out = prod.set_index('model').loc[scores.index, ['brand', 'price_tier', 'avg_price', 'avg_rating']].copy()
    out['similarity'] = scores.round(3)
    return out.reset_index()