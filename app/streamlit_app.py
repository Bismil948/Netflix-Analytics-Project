import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv('data/netflix_titles.csv')
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
df.drop_duplicates(inplace=True)
df['description'] = df['description'].fillna('')

st.sidebar.header("Filters")

countries = ["All"] + sorted(df['country'].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Select Country", countries)

genres = sorted({g.strip() for cell in df['listed_in'].dropna() for g in cell.split(',')})
selected_genres = st.sidebar.multiselect("Select Genre(s)", genres)

types = ["All"] + sorted(df['type'].dropna().unique().tolist())
selected_type = st.sidebar.selectbox("Select Type", types)

year_min, year_max = int(df['release_year'].min()), int(df['release_year'].max())
selected_years = st.sidebar.slider("Select Release Year Range", min_value=year_min, max_value=year_max, value=(year_min, year_max))

mask = pd.Series(True, index=df.index)
if selected_country != "All":
    mask &= df['country'].fillna('').str.contains(selected_country)
if selected_genres:
    mask &= df['listed_in'].apply(lambda s: all(g in s for g in selected_genres) if isinstance(s, str) else False)
if selected_type != "All":
    mask &= df['type'] == selected_type
mask &= df['release_year'].between(selected_years[0], selected_years[1])
filtered = df[mask]

st.title("🎬 Netflix Shows Analytics Dashboard")

total_titles = len(filtered)
movies = (filtered['type'].str.lower() == 'movie').sum()
tv_shows = (filtered['type'].str.lower() == 'tv show').sum()
earliest_year = int(filtered['release_year'].min()) if not filtered.empty else 0
latest_year = int(filtered['release_year'].max()) if not filtered.empty else 0

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Titles", total_titles)
kpi2.metric("Movies", movies)
kpi3.metric("TV Shows", tv_shows)
kpi4.metric("Earliest Year", earliest_year)
kpi5.metric("Latest Year", latest_year)

genre_counts = filtered['listed_in'].str.split(',').explode().str.strip().value_counts().nlargest(10)
fig_genre = px.bar(genre_counts, x=genre_counts.values, y=genre_counts.index, orientation='h', labels={'x':'# Titles','y':'Genre'})
st.subheader("Top Genres")
st.plotly_chart(fig_genre)

yearly = filtered.groupby('release_year').size()
fig_year = px.line(yearly, x=yearly.index, y=yearly.values, labels={'x':'Year','y':'# Titles'})
st.subheader("Releases Over Time")
st.plotly_chart(fig_year)

rating_counts = filtered['rating'].value_counts()
fig_rating = px.pie(rating_counts, names=rating_counts.index, values=rating_counts.values)
st.subheader("Rating Distribution")
st.plotly_chart(fig_rating)

cg = filtered.assign(genres=filtered['listed_in'].str.split(',')).explode('genres')
cg['genres'] = cg['genres'].str.strip()
top_countries = cg['country'].value_counts().nlargest(10).index
heatdata = cg[cg['country'].isin(top_countries)].pivot_table(index='country', columns='genres', aggfunc='size', fill_value=0)
fig_heat = px.imshow(heatdata, aspect="auto", labels=dict(x="Genre", y="Country", color="# Titles"))
st.subheader("Country × Genre Heatmap")
st.plotly_chart(fig_heat)
st.subheader("Top Titles (Filtered)")
st.dataframe(filtered[['title','type','country','release_year','rating','listed_in']].sort_values('release_year', ascending=False).head(50))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['description'])

def recommend(title, df, tfidf_matrix):
    idx = df[df['title']==title].index[0]
    cosine_sim = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    sim_idx = cosine_sim.argsort()[::-1][1:6]
    return df.iloc[sim_idx][['title','type','listed_in','country','release_year']]

st.subheader("Recommendations")
selected_title = st.selectbox("Pick a title to get recommendations", options=df['title'].sample(200).sort_values())
if selected_title:
    recs = recommend(selected_title, df, tfidf_matrix)
    st.write(recs)

