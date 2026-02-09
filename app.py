import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Childhood Cartoons Archive", layout="wide")

# ---------------------- PRETTY COLUMNS ----------------------
PRETTY_COLS = {
    "title": "Title",
    "network_norm": "Network",
    "year_start": "Start Year",
    "decade": "Decade",
    "tmdb_vote_average": "Rating",
    "tmdb_vote_count": "Votes",
    "tmdb_popularity": "Popularity",
    "tmdb_number_of_seasons": "Seasons",
    "tmdb_number_of_episodes": "Episodes",
    "tmdb_genres": "Genres",
    "tmdb_detected_type": "TMDB Type",
    "tmdb_error": "TMDB Error",
    "similarity": "Similarity",
}

DISPLAY_COLS = [
    "title",
    "network_norm",
    "year_start",
    "decade",
    "tmdb_genres",
    "tmdb_vote_average",
    "tmdb_vote_count",
    "tmdb_number_of_seasons",
    "tmdb_number_of_episodes",
    "tmdb_popularity",
]

def pretty_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: PRETTY_COLS.get(c, c) for c in df.columns})

def select_display_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    return df[cols].copy()

# ---------------------- LOAD DATA ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cartoons_clean.csv")

    # numeric columns
    for col in [
        "tmdb_vote_average","tmdb_vote_count","tmdb_number_of_episodes",
        "tmdb_number_of_seasons","tmdb_popularity","decade","year_start"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # genres as list
    if "tmdb_genres" in df.columns:
        df["genres_list"] = df["tmdb_genres"].fillna("").apply(
            lambda s: [g.strip() for g in str(s).split(",") if g.strip()]
        )
    else:
        df["genres_list"] = [[] for _ in range(len(df))]

    # helper for search
    if "title" in df.columns:
        df["title_lower"] = df["title"].astype(str).str.lower()

    return df

df = load_data()

# ---------------------- RECOMMENDER ----------------------
def build_profile_text(row):
    genres = str(row.get("tmdb_genres", "") or "")
    network = str(row.get("network_norm", "") or "")
    decade = str(row.get("decade", "") or "")
    # extra weight for genres + network
    return f"{genres} {genres} {network} {network} decade_{decade}"

@st.cache_data
def build_recommender(df_in: pd.DataFrame):
    df2 = df_in.copy()

    for c in ["title", "tmdb_genres", "network_norm", "decade"]:
        if c not in df2.columns:
            df2[c] = ""

    df2["profile_text"] = df2.apply(build_profile_text, axis=1)

    vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b[\w&]+\b")
    X = vec.fit_transform(df2["profile_text"].fillna(""))

    sim = cosine_similarity(X, X)
    title_to_idx = {t: i for i, t in enumerate(df2["title"].astype(str))}

    return df2, sim, title_to_idx

rec_df, sim_matrix, title_to_idx = build_recommender(df)

def build_explanation(selected_title: str, idx: int, rec_df: pd.DataFrame, sim_matrix):
    """
    Build ONE general explanation text based on the top similar items.
    """
    base_row = rec_df.iloc[idx]

    # take top 8 to extract patterns
    scores_preview = list(enumerate(sim_matrix[idx]))
    scores_preview = [(i, s) for i, s in scores_preview if i != idx]
    scores_preview.sort(key=lambda x: x[1], reverse=True)

    preview_idx = [i for i, _ in scores_preview[:8]]
    preview_rows = rec_df.iloc[preview_idx]

    # frequent common genres
    base_genres = [g.strip() for g in str(base_row.get("tmdb_genres", "") or "").split(",") if g.strip()]
    base_set = set(base_genres)

    freq = {}
    for _, r in preview_rows.iterrows():
        g2 = set([g.strip() for g in str(r.get("tmdb_genres", "") or "").split(",") if g.strip()])
        common = base_set.intersection(g2)
        for g in common:
            freq[g] = freq.get(g, 0) + 1

    common_genres_sorted = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    common_genres = [g for g, _ in common_genres_sorted[:3]]

    network = str(base_row.get("network_norm", "") or "").strip()
    decade = base_row.get("decade", None)

    parts = []
    if common_genres:
        parts.append(f"shares genres like **{', '.join(common_genres)}**")
    if network:
        parts.append(f"it is associated with **{network}**")
    if pd.notna(decade):
        try:
            parts.append(f"and it belongs to the same era (**{int(decade)}s**)")
        except Exception:
            pass

    if not parts:
        return "Recommendations are generated from similar metadata (genre, network and era)."

    text = ", ".join(parts[:-1]) + (" " + parts[-1] if len(parts) > 1 else parts[0])
    return f"Recommendations appear because **{selected_title}** {text}."

# ---------------------- UI ----------------------
st.title("📺 Childhood Cartoons Archive")
st.caption("Custom dataset built from Trakt lists and enriched with TMDB metadata (cleaned & normalized).")

tab_dash, tab_rank, tab_data, tab_rec = st.tabs(["📊 Overview", "🏆 Rankings", "🗂️ Dataset", "🤝 Recommendations"])

# ---------------- Sidebar ----------------
st.sidebar.header("Filters")

# Network
networks = sorted(df["network_norm"].dropna().unique()) if "network_norm" in df.columns else []
selected_networks = st.sidebar.multiselect("Network", options=networks, default=networks)

# Years
min_year = int(df["year_start"].dropna().min()) if "year_start" in df.columns and df["year_start"].notna().any() else 1950
max_year = int(df["year_start"].dropna().max()) if "year_start" in df.columns and df["year_start"].notna().any() else 2030
year_range = st.sidebar.slider("Start year", min_value=min_year, max_value=max_year, value=(min_year, max_year))

# Rating
vote_range = st.sidebar.slider("TMDB rating", 0.0, 10.0, (0.0, 10.0))

# Search
query = st.sidebar.text_input("Search title", placeholder="e.g., ben 10, gumball...").strip().lower()

# Genres
all_genres = sorted({g for gs in df["genres_list"] for g in (gs or [])})
selected_genres = st.sidebar.multiselect("Genres", options=all_genres)

# ---------------- Apply filters ----------------
filtered = df.copy()

if selected_networks and "network_norm" in filtered.columns:
    filtered = filtered[filtered["network_norm"].isin(selected_networks)]

if "year_start" in filtered.columns:
    filtered = filtered[
        (filtered["year_start"].isna()) |
        ((filtered["year_start"] >= year_range[0]) & (filtered["year_start"] <= year_range[1]))
    ]

if "tmdb_vote_average" in filtered.columns:
    filtered = filtered[
        (filtered["tmdb_vote_average"].isna()) |
        ((filtered["tmdb_vote_average"] >= vote_range[0]) & (filtered["tmdb_vote_average"] <= vote_range[1]))
    ]

if query and "title_lower" in filtered.columns:
    filtered = filtered[filtered["title_lower"].str.contains(query, na=False)]

if selected_genres:
    filtered = filtered[filtered["genres_list"].apply(lambda gs: any(g in (gs or []) for g in selected_genres))]

# Download filtered CSV (main columns only)
csv_bytes = pretty_table(select_display_cols(filtered)).to_csv(index=False).encode("utf-8")
st.sidebar.download_button("⬇️ Download filtered CSV", csv_bytes, "cartoons_filtered.csv", "text/csv")

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Titles", len(filtered))
k2.metric("Unique networks", filtered["network_norm"].nunique() if "network_norm" in filtered.columns else 0)
k3.metric("Avg rating", f"{filtered['tmdb_vote_average'].mean():.2f}" if "tmdb_vote_average" in filtered.columns else "-")
k4.metric("Avg episodes", f"{filtered['tmdb_number_of_episodes'].mean():.0f}" if "tmdb_number_of_episodes" in filtered.columns else "-")

# ---------------- OVERVIEW ----------------
with tab_dash:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Cartoons by decade")
        if "decade" in filtered.columns and filtered["decade"].notna().any():
            counts = filtered.dropna(subset=["decade"]).groupby("decade")["title"].count().sort_index()
            fig = plt.figure()
            plt.bar(counts.index.astype(int).astype(str), counts.values)
            plt.xticks(rotation=45)
            plt.ylabel("Count")
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("Not enough decade data.")

    with c2:
        st.subheader("Top networks (count)")
        if "network_norm" in filtered.columns and filtered["network_norm"].notna().any():
            topn = (
                filtered.dropna(subset=["network_norm"])
                .groupby("network_norm")["title"]
                .count()
                .sort_values(ascending=False)
                .head(10)
            )
            fig = plt.figure()
            plt.bar(topn.index, topn.values)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Count")
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("No network data available.")

    st.divider()

    st.subheader("Episodes vs Rating (TMDB)")
    if "tmdb_number_of_episodes" in filtered.columns and "tmdb_vote_average" in filtered.columns:
        plot_df = filtered.dropna(subset=["tmdb_number_of_episodes", "tmdb_vote_average"])
        if len(plot_df) == 0:
            st.info("Not enough points to plot.")
        else:
            fig = plt.figure()
            plt.scatter(plot_df["tmdb_number_of_episodes"], plot_df["tmdb_vote_average"])
            plt.xlabel("Number of episodes")
            plt.ylabel("Average rating (TMDB)")
            st.pyplot(fig, clear_figure=True)

    st.divider()

    st.subheader("Most common genres")
    genre_counts = {}
    for gs in filtered["genres_list"]:
        for g in (gs or []):
            genre_counts[g] = genre_counts.get(g, 0) + 1

    if genre_counts:
        gdf = (
            pd.DataFrame({"genre": list(genre_counts.keys()), "count": list(genre_counts.values())})
            .sort_values("count", ascending=False)
            .head(12)
        )
        fig = plt.figure()
        plt.bar(gdf["genre"], gdf["count"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Count")
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("No genre data to count.")

# ---------------- RANKINGS ----------------
with tab_rank:
    st.subheader("Top by rating (TMDB)")
    top_score = (
        filtered.dropna(subset=["tmdb_vote_average"])
        .sort_values(["tmdb_vote_average", "tmdb_vote_count"], ascending=False)
        .head(20)
    )
    st.dataframe(pretty_table(select_display_cols(top_score)), use_container_width=True)

    st.subheader("Top by popularity (TMDB)")
    top_pop = (
        filtered.dropna(subset=["tmdb_popularity"])
        .sort_values("tmdb_popularity", ascending=False)
        .head(20)
    )
    st.dataframe(pretty_table(select_display_cols(top_pop)), use_container_width=True)

    st.subheader("Top by number of episodes")
    top_eps = (
        filtered.dropna(subset=["tmdb_number_of_episodes"])
        .sort_values("tmdb_number_of_episodes", ascending=False)
        .head(20)
    )
    st.dataframe(pretty_table(select_display_cols(top_eps)), use_container_width=True)

# ---------------- DATASET ----------------
with tab_data:
    st.subheader("Filtered dataset (main columns)")
    st.dataframe(pretty_table(select_display_cols(filtered)), use_container_width=True)

# ---------------- RECOMMENDATIONS ----------------
with tab_rec:
    st.subheader("🤝 Cartoon recommender")
    st.caption("Select a show and see similar cartoons based on genre, network and era.")

    avail_titles = sorted(filtered["title"].astype(str).unique().tolist())
    if not avail_titles:
        st.info("No titles available with the current filters.")
    else:
        selected_title = st.selectbox("Pick a cartoon", avail_titles)
        k = st.slider("How many suggestions?", min_value=3, max_value=20, value=10)

        if selected_title not in title_to_idx:
            st.error("Title not found in the dataset.")
        else:
            idx = title_to_idx[selected_title]

            explanation = build_explanation(selected_title, idx, rec_df, sim_matrix)
            st.info(explanation)

            scores = list(enumerate(sim_matrix[idx]))
            scores = [(i, s) for i, s in scores if i != idx]
            scores.sort(key=lambda x: x[1], reverse=True)

            top = scores[: max(k * 5, 30)]
            top_idx = [i for i, _ in top]

            recs = rec_df.iloc[top_idx].copy()
            recs["similarity"] = [s for _, s in top]

            # keep consistency with current filters
            allowed_titles = set(filtered["title"].astype(str).tolist())
            recs = recs[recs["title"].astype(str).isin(allowed_titles)].head(k)

            show_cols = [c for c in [
                "title", "network_norm", "year_start", "decade", "tmdb_genres",
                "tmdb_vote_average", "tmdb_vote_count", "tmdb_number_of_episodes",
                "tmdb_popularity", "similarity"
            ] if c in recs.columns]

            st.dataframe(pretty_table(recs[show_cols]), use_container_width=True)