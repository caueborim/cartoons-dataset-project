# 📺 Childhood Cartoons Dataset (Data Engineering + Analysis Project)

## Live Dashboard
🔗 https://cartoon-data-dashboard.streamlit.app/

A personal data project that transforms a nostalgic list of childhood cartoons into a structured and enriched dataset using public APIs.

## Problem

I had a personal list of cartoons I watched growing up (Cartoon Network, Nickelodeon, Disney, Fox Kids/Jetix and Brazilian TV).
However, it existed only as memory — not analyzable data.

The challenge:

> How can we convert a subjective memory-based list into a real dataset suitable for analysis and visualization?

## Solution

I built a complete data pipeline using two public APIs:

* **Trakt API** → extract list items
* **TMDB API** → enrich metadata (genres, networks, ratings, episodes, popularity)

Then cleaned, normalized and analyzed the data, and built an interactive dashboard.

## Pipeline

1. Extract lists from Trakt
2. Convert to structured CSV
3. Enrich using TMDB metadata
4. Handle missing matches (override system)
5. Clean and normalize fields
6. Create visual dashboard (Streamlit)

## Dataset Features

Each show includes:

* Start year
* Decade
* Genres
* Network
* Number of seasons & episodes
* TMDB rating
* Popularity
* Language

## Tools Used

* Python
* Pandas
* Requests
* Streamlit
* Public REST APIs
* Data cleaning & transformation

## Dashboard

Interactive dashboard with:

* Filters by network, year and rating
* Rankings
* Popularity vs rating analysis
* Episodes vs score correlation
* Decade distribution

## Key Insight

This project demonstrates how unstructured personal information (memories) can be transformed into a structured dataset and explored with data analysis techniques.

## How to Run

1. Create a `.env` file using `.env.example`
2. Add your Trakt and TMDB API keys
3. Run data pipeline:

```
python export_trakt_lists.py
python enrich_tmdb.py
python prepare_dataset.py
```

4. Launch dashboard:

```
streamlit run app.py
```

---

This project focuses on data collection, data cleaning, API integration, and exploratory data analysis.
