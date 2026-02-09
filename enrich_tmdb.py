import os
import time
from typing import Dict, Optional, List, Tuple

import requests
import pandas as pd

TMDB_BASE = "https://api.themoviedb.org/3"


def load_env(path: str = ".env") -> None:
    """Carrega .env e sobrescreve variáveis."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()


def tmdb_get(
    session: requests.Session,
    api_key: str,
    path: str,
    params: Optional[Dict] = None,
    retries: int = 5,
):
    """
    GET TMDB com retry.
    Se receber 404 -> retorna None (não quebra o pipeline).
    """
    if params is None:
        params = {}
    params = dict(params)
    params["api_key"] = api_key

    url = f"{TMDB_BASE}{path}"
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params=params, timeout=45)

            if r.status_code == 404:
                return None

            if r.status_code in (429, 500, 502, 503, 504):
                wait = min(2 ** attempt, 20)
                print(f"[WARN] TMDB {r.status_code} em {path}. Retry em {wait}s...")
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r.json()

        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            wait = min(2 ** attempt, 20)
            print(f"[WARN] TMDB timeout/conexão (tentativa {attempt}/{retries}) em {path}. Esperando {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Falha TMDB em {path}: {last_err}")


def pick_network_and_country_tv(details: Dict) -> Tuple[Optional[str], Optional[str]]:
    networks = details.get("networks") or []
    network = networks[0].get("name") if isinstance(networks, list) and networks else None

    origin = details.get("origin_country") or []
    country = origin[0] if isinstance(origin, list) and origin else None

    return network, country


def main():
    load_env()
    tmdb_key = os.getenv("TMDB_API_KEY")
    if not tmdb_key:
        raise SystemExit("Faltou TMDB_API_KEY no .env")

    if not os.path.exists("cartoons_trakt.csv"):
        raise SystemExit("Não achei cartoons_trakt_fixed.csv na pasta. Rode o export_trakt_lists.py primeiro.")

    df = pd.read_csv("cartoons_trakt_fixed.csv")

    if "tmdb_id" not in df.columns:
        raise SystemExit("Seu CSV não tem tmdb_id.")

    session = requests.Session()

    enriched_rows: List[Dict] = []
    problems: List[Dict] = []

    total = len(df)
    for i, row in df.iterrows():
        tmdb_id = row.get("tmdb_id")
        trakt_type = row.get("type")
        title = row.get("title")

        # Se não tem TMDB ID, segue sem quebrar
        if pd.isna(tmdb_id):
            enriched_rows.append({"tmdb_error": "missing_tmdb_id"})
            problems.append(
                {
                    "row_index": int(i),
                    "title": title,
                    "trakt_type": trakt_type,
                    "tmdb_id": None,
                    "problem": "missing_tmdb_id",
                }
            )
            continue

        tmdb_id = int(tmdb_id)

        # 1) tenta pelo tipo do Trakt
        details = None
        det_type = None  # "tv" ou "movie"

        if trakt_type == "movie":
            details = tmdb_get(session, tmdb_key, f"/movie/{tmdb_id}", params={"language": "en-US"})
            det_type = "movie"
        else:
            details = tmdb_get(session, tmdb_key, f"/tv/{tmdb_id}", params={"language": "en-US"})
            det_type = "tv"

        # 2) fallback: se deu 404, tenta o outro endpoint
        if details is None:
            if det_type == "tv":
                details = tmdb_get(session, tmdb_key, f"/movie/{tmdb_id}", params={"language": "en-US"})
                det_type = "movie"
            else:
                details = tmdb_get(session, tmdb_key, f"/tv/{tmdb_id}", params={"language": "en-US"})
                det_type = "tv"

        # 3) se ainda não achou, registra e segue
        if details is None:
            print(f"[SKIP] TMDB id {tmdb_id} não encontrado em /tv nem /movie. Seguindo...")
            enriched_rows.append({"tmdb_error": "not_found", "tmdb_detected_type": None})
            problems.append(
                {
                    "row_index": int(i),
                    "title": title,
                    "trakt_type": trakt_type,
                    "tmdb_id": tmdb_id,
                    "problem": "not_found_tv_or_movie",
                }
            )
            continue

        # Normaliza campos (dependendo do tipo detectado)
        genres = [g.get("name") for g in (details.get("genres") or []) if isinstance(g, dict)]
        vote_avg = details.get("vote_average")
        vote_count = details.get("vote_count")
        popularity = details.get("popularity")
        original_lang = details.get("original_language")
        status = details.get("status")

        if det_type == "movie":
            tmdb_name = details.get("title")
            first_air = details.get("release_date")
            runtime = details.get("runtime")
            network, country = None, None
            seasons, episodes = None, None
        else:
            tmdb_name = details.get("name")
            first_air = details.get("first_air_date")
            runtime_list = details.get("episode_run_time") or []
            runtime = runtime_list[0] if isinstance(runtime_list, list) and runtime_list else None
            seasons = details.get("number_of_seasons")
            episodes = details.get("number_of_episodes")
            network, country = pick_network_and_country_tv(details)

        enriched_rows.append(
            {
                "tmdb_detected_type": det_type,
                "tmdb_name": tmdb_name,
                "tmdb_first_air_date": first_air,
                "tmdb_status": status,
                "tmdb_genres": ", ".join([g for g in genres if g]),
                "tmdb_runtime_min": runtime,
                "tmdb_network": network,
                "tmdb_origin_country": country,
                "tmdb_number_of_seasons": seasons,
                "tmdb_number_of_episodes": episodes,
                "tmdb_vote_average": vote_avg,
                "tmdb_vote_count": vote_count,
                "tmdb_popularity": popularity,
                "tmdb_original_language": original_lang,
                "tmdb_error": None,
            }
        )

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"[OK] Enriquecidos {i+1}/{total}")

        time.sleep(0.15)

    enrich_df = pd.DataFrame(enriched_rows)
    out = pd.concat([df.reset_index(drop=True), enrich_df.reset_index(drop=True)], axis=1)

    # salva outputs
    out.to_csv("cartoons_enriched.csv", index=False, encoding="utf-8")
    out.to_json("cartoons_enriched.json", orient="records", force_ascii=False, indent=2)

    if problems:
        pd.DataFrame(problems).to_csv("tmdb_problems.csv", index=False, encoding="utf-8")

    print("\n✅ Finalizado!")
    print(f"- cartoons_enriched.csv  (linhas: {len(out)})")
    print(f"- cartoons_enriched.json (linhas: {len(out)})")
    print(f"- tmdb_problems.csv (problemas: {len(problems)})")


if __name__ == "__main__":
    main()