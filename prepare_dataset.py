import pandas as pd
import ast

df = pd.read_csv("cartoons_enriched.csv")

# -----------------------
# Datas
# -----------------------
df["tmdb_first_air_date"] = pd.to_datetime(df["tmdb_first_air_date"], errors="coerce")
df["year_start"] = df["tmdb_first_air_date"].dt.year.fillna(df["year"]).astype("Int64")

# Décadas
df["decade"] = (df["year_start"] // 10 * 10).astype("Int64")

# -----------------------
# Network normalizada
# -----------------------
def norm_network(x):
    if pd.isna(x):
        return None
    x = str(x).strip()

    mapping = {
        "Cartoon Network": "Cartoon Network",
        "Nickelodeon": "Nickelodeon",
        "Disney Channel": "Disney Channel",
        "Disney XD": "Disney XD",
        "Adult Swim": "Adult Swim",
        "Fox": "Fox Kids/Jetix",
        "The WB": "Kids WB",
    }

    return mapping.get(x, x)

df["network_norm"] = df["tmdb_network"].apply(norm_network)

# -----------------------
# Gêneros (CORREÇÃO AQUI)
# -----------------------
def parse_genres(s):
    if pd.isna(s):
        return []

    s = str(s)

    # se já vier tipo lista em string
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except:
            pass

    # fallback: split normal
    return [g.strip() for g in s.split(",") if g.strip()]

df["genres_list"] = df["tmdb_genres"].apply(parse_genres)

# -----------------------
# Flag erro
# -----------------------
df["has_tmdb_error"] = df["tmdb_error"].notna()

df.to_csv("cartoons_clean.csv", index=False, encoding="utf-8")

print("✅ cartoons_clean.csv recriado corretamente")