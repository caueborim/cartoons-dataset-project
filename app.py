import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Arquivo das Animações da Infância", layout="wide")

# ---------------------- COLUNAS BONITAS ----------------------
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

    # numéricos
    for col in [
        "tmdb_vote_average","tmdb_vote_count","tmdb_number_of_episodes",
        "tmdb_number_of_seasons","tmdb_popularity","decade","year_start"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # gêneros como lista
    if "tmdb_genres" in df.columns:
        df["genres_list"] = df["tmdb_genres"].fillna("").apply(
            lambda s: [g.strip() for g in str(s).split(",") if g.strip()]
        )
    else:
        df["genres_list"] = [[] for _ in range(len(df))]

    # helper search
    if "title" in df.columns:
        df["title_lower"] = df["title"].astype(str).str.lower()

    return df


df = load_data()

st.title("📺 Arquivo das Animações da Infância")
st.caption("Dataset autoral: Trakt + TMDB + limpeza e normalização de dados")

tab_dash, tab_rank, tab_data = st.tabs(["📊 Dashboard", "🏆 Rankings", "🗂️ Dados"])

# ---------------- Sidebar ----------------
st.sidebar.header("Filtros")

# Network
networks = sorted(df["network_norm"].dropna().unique()) if "network_norm" in df.columns else []
selected_networks = st.sidebar.multiselect("Network", options=networks, default=networks)

# Ano
min_year = int(df["year_start"].dropna().min()) if "year_start" in df.columns and df["year_start"].notna().any() else 1950
max_year = int(df["year_start"].dropna().max()) if "year_start" in df.columns and df["year_start"].notna().any() else 2030
year_range = st.sidebar.slider("Ano de início", min_value=min_year, max_value=max_year, value=(min_year, max_year))

# Nota
vote_range = st.sidebar.slider("Nota TMDB", 0.0, 10.0, (0.0, 10.0))

# Busca por título
query = st.sidebar.text_input("Buscar título", placeholder="ex: ben 10, gumball...").strip().lower()

# Gêneros
all_genres = sorted({g for gs in df["genres_list"] for g in (gs or [])})
selected_genres = st.sidebar.multiselect("Gêneros", options=all_genres)

# ---------------- Filtros ----------------
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

# Download CSV filtrado (SÓ COLUNAS BOAS)
csv_bytes = pretty_table(select_display_cols(filtered)).to_csv(index=False).encode("utf-8")
st.sidebar.download_button("⬇️ Baixar CSV filtrado", csv_bytes, "cartoons_filtered.csv", "text/csv")

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Títulos", len(filtered))
k2.metric("Networks", filtered["network_norm"].nunique() if "network_norm" in filtered.columns else 0)
k3.metric("Nota média", f"{filtered['tmdb_vote_average'].mean():.2f}" if "tmdb_vote_average" in filtered.columns else "-")
k4.metric("Episódios médios", f"{filtered['tmdb_number_of_episodes'].mean():.0f}" if "tmdb_number_of_episodes" in filtered.columns else "-")

# ---------------- DASHBOARD ----------------
with tab_dash:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Cartoons por década")
        if "decade" in filtered.columns and filtered["decade"].notna().any():
            counts = filtered.dropna(subset=["decade"]).groupby("decade")["title"].count().sort_index()
            fig = plt.figure()
            plt.bar(counts.index.astype(int).astype(str), counts.values)
            plt.xticks(rotation=45)
            plt.ylabel("Quantidade")
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("Sem dados suficientes de década.")

    with c2:
        st.subheader("Top networks (quantidade)")
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
            plt.ylabel("Quantidade")
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("Sem dados de network.")

    st.divider()

    st.subheader("Episódios vs Nota (TMDB)")
    if "tmdb_number_of_episodes" in filtered.columns and "tmdb_vote_average" in filtered.columns:
        plot_df = filtered.dropna(subset=["tmdb_number_of_episodes", "tmdb_vote_average"])
        if len(plot_df) == 0:
            st.info("Sem pontos suficientes para plotar.")
        else:
            fig = plt.figure()
            plt.scatter(plot_df["tmdb_number_of_episodes"], plot_df["tmdb_vote_average"])
            plt.xlabel("Número de episódios")
            plt.ylabel("Nota média (TMDB)")
            st.pyplot(fig, clear_figure=True)

    st.divider()

    st.subheader("Gêneros mais comuns")
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
        plt.ylabel("Quantidade")
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Sem gêneros para contar.")

# ---------------- RANKINGS ----------------
with tab_rank:
    st.subheader("Top por Nota (TMDB)")
    top_score = (
        filtered.dropna(subset=["tmdb_vote_average"])
        .sort_values(["tmdb_vote_average", "tmdb_vote_count"], ascending=False)
        .head(20)
    )
    st.dataframe(pretty_table(select_display_cols(top_score)), use_container_width=True)

    st.subheader("Top por Popularidade (TMDB)")
    top_pop = (
        filtered.dropna(subset=["tmdb_popularity"])
        .sort_values("tmdb_popularity", ascending=False)
        .head(20)
    )
    st.dataframe(pretty_table(select_display_cols(top_pop)), use_container_width=True)

    st.subheader("Top por Número de Episódios")
    top_eps = (
        filtered.dropna(subset=["tmdb_number_of_episodes"])
        .sort_values("tmdb_number_of_episodes", ascending=False)
        .head(20)
    )
    st.dataframe(pretty_table(select_display_cols(top_eps)), use_container_width=True)

# ---------------- DATA ----------------
with tab_data:
    st.subheader("Dataset filtrado (colunas principais)")
    st.dataframe(pretty_table(select_display_cols(filtered)), use_container_width=True)