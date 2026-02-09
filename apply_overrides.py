import pandas as pd

df = pd.read_csv("cartoons_trakt.csv")
ov = pd.read_csv("tmdb_overrides.csv")

m = df.merge(ov, on="title", how="left")

mask = m["tmdb_id_override"].notna()
m.loc[mask, "tmdb_id"] = m.loc[mask, "tmdb_id_override"].astype(int)

m = m.drop(columns=["tmdb_id_override", "tmdb_type_override"])

m.to_csv("cartoons_trakt_fixed.csv", index=False, encoding="utf-8")

print("✅ Gerado: cartoons_trakt_fixed.csv (com overrides aplicados)")
print("Overridden:", int(mask.sum()))