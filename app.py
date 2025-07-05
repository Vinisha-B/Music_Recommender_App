import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("SpotifyFeatures.csv")
df.drop_duplicates(subset='track_name', inplace=True)
df.reset_index(drop=True, inplace=True)

# Normalize features
features = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

# Recommendation function
def get_recommendations(song_title):
    try:
        idx = df[df['track_name'].str.lower() == song_title.lower()].index[0]
    except IndexError:
        return None
    sim_scores = cosine_similarity([df_scaled.loc[idx]], df_scaled)
    similar_indices = sim_scores[0].argsort()[::-1][1:11]
    return df.iloc[similar_indices][['track_name', 'artist_name', 'genre', 'valence', 'energy', 'tempo']]

# Streamlit UI
# Streamlit App
st.title("🎵 Spotify Music Recommender")

mode = st.radio("Choose how you'd like to get recommendations:", ["By Song", "By Mood"])

if mode == "By Song":
    song_input = st.text_input("Enter a song you like")

    if song_input:
        recommendations = get_recommendations(song_input)
        if recommendations is not None:
            st.subheader("🔁 Recommended Songs")
            st.dataframe(recommendations)
        else:
            st.warning("⚠️ Song not found. Please check the name or try a different one.")

elif mode == "By Mood":
    mood = st.selectbox("Pick a Mood", ["Happy 😊", "Sad 😔", "Energetic 💃"])
    
    if mood == "Happy 😊":
        filtered = df[(df['valence'] > 0.6) & (df['energy'] > 0.5)]
    elif mood == "Sad 😔":
        filtered = df[(df['valence'] < 0.4) & (df['energy'] < 0.5)]
    elif mood == "Energetic 💃":
        filtered = df[df['energy'] > 0.75]

    top_mood_songs = filtered[['track_name', 'artist_name', 'genre', 'valence', 'energy', 'tempo']].sample(10)
    st.subheader(f"🎶 {mood} Songs")
    st.dataframe(top_mood_songs)
