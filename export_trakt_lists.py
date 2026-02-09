import os
import time
from typing import Dict, List, Optional

import requests
import pandas as pd

USER = "cauemborim"
LIST_SLUGS = [
    "arquivo-dos-desenhos-da-infancia",
    "arquivo-dos-desenhos-da-infancia-2",
]

API_BASE = "https://api.trakt.tv"
API_VERSION = "2"


def load_env(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()  # sobrescreve sempre


def trakt_headers(client_id: str) -> Dict[str, str]:
    return {
        "trakt-api-version": API_VERSION,
        "trakt-api-key": client_id,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }


def get_user_lists(client_id: str, user: str) -> List[dict]:
    r = requests.get(
        f"{API_BASE}/users/{user}/lists",
        headers=trakt_headers(client_id),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def slug_to_list_id(user_lists: List[dict], slug: str) -> int:
    for lst in user_lists:
        ids = lst.get("ids") or {}
        if ids.get("slug") == slug and isinstance(ids.get("trakt"), int):
            return ids["trakt"]
    raise RuntimeError(f"Não achei o ID numérico para o slug: {slug}")


def fetch_list_items_by_id(client_id: str, list_id: int) -> List[dict]:
    """Baixa itens via /lists/{id}/items (mais confiável)."""
    headers = trakt_headers(client_id)
    session = requests.Session()

    items: List[dict] = []
    page = 1
    limit = 50
    timeout_s = 90

    while True:
        url = f"{API_BASE}/lists/{list_id}/items"
        params = {"extended": "min", "page": page, "limit": limit}

        batch = None
        last_err = None

        for attempt in range(1, 6):
            try:
                r = session.get(url, headers=headers, params=params, timeout=timeout_s)

                if r.status_code == 401:
                    raise RuntimeError("401 Unauthorized em /lists/{id}/items (estranho). Confira TRAKT_CLIENT_ID.")
                if r.status_code == 404:
                    raise RuntimeError(f"404 Not Found: lista id={list_id} não existe")
                if r.status_code in (429, 500, 502, 503, 504):
                    wait = min(2 ** attempt, 20)
                    print(f"[WARN] HTTP {r.status_code} (page {page}). Retry em {wait}s...")
                    time.sleep(wait)
                    continue

                r.raise_for_status()
                batch = r.json()
                last_err = None
                break

            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                last_err = e
                wait = min(2 ** attempt, 20)
                print(f"[WARN] Timeout/conexão (tentativa {attempt}/5) page {page}. Esperando {wait}s...")
                time.sleep(wait)

        if last_err is not None:
            raise RuntimeError(f"Falhou ao baixar page {page}. Erro: {last_err}")

        if not batch:
            break

        items.extend(batch)
        page += 1
        time.sleep(0.15)

    return items


def extract_core(obj: object):
    if not isinstance(obj, dict):
        return None, None, {}
    title = obj.get("title")
    year = obj.get("year")
    ids = obj.get("ids") if isinstance(obj.get("ids"), dict) else {}
    return title, year, ids


def flatten_items(raw_items: List[dict], source_slug: str, source_list_id: int) -> pd.DataFrame:
    rows = []
    for it in raw_items:
        typ = it.get("type")  # show/movie
        obj = it.get(typ) if isinstance(it, dict) else None

        title, year, ids = extract_core(obj)

        rows.append(
            {
                "source_list_slug": source_slug,
                "source_list_id": source_list_id,
                "rank": it.get("rank"),
                "listed_at": it.get("listed_at"),
                "list_item_id": it.get("id"),
                "type": typ,
                "title": title,
                "year": year,
                "trakt_id": ids.get("trakt"),
                "trakt_slug": ids.get("slug"),
                "tmdb_id": ids.get("tmdb"),
                "imdb_id": ids.get("imdb"),
                "tvdb_id": ids.get("tvdb"),
            }
        )

    df = pd.DataFrame(rows)
    if "type" in df.columns:
        df = df[df["type"].isin(["show", "movie"])].copy()
    return df


def main():
    print("=== EXPORT TRAKT v6 (LIST ID) ===")

    os.environ.pop("TRAKT_CLIENT_ID", None)
    load_env()
    client_id = os.getenv("TRAKT_CLIENT_ID")
    if not client_id:
        raise SystemExit("Faltou TRAKT_CLIENT_ID no .env")

    # teste rápido
    test = requests.get(
        "https://api.trakt.tv/movies/popular",
        headers=trakt_headers(client_id),
        timeout=30,
    )
    print("[TEST] movies/popular status:", test.status_code)
    if test.status_code != 200:
        raise SystemExit("Client ID não está funcionando. Confira o .env.")

    user_lists = get_user_lists(client_id, USER)

    all_dfs = []
    for slug in LIST_SLUGS:
        list_id = slug_to_list_id(user_lists, slug)
        print(f"[INFO] Baixando por ID: {slug} -> {list_id}")

        raw = fetch_list_items_by_id(client_id, list_id)
        print(f"[OK] Itens: {len(raw)}")

        all_dfs.append(flatten_items(raw, slug, list_id))

    big = pd.concat(all_dfs, ignore_index=True)

    # remove duplicados entre as duas listas
    if big["trakt_id"].notna().any():
        big = big.drop_duplicates(subset=["trakt_id"], keep="first")
    else:
        big = big.drop_duplicates(subset=["trakt_slug", "title"], keep="first")

    big = big.sort_values(["type", "title"], na_position="last").reset_index(drop=True)

    big.to_csv("cartoons_trakt.csv", index=False, encoding="utf-8")
    big.to_json("cartoons_trakt.json", orient="records", force_ascii=False, indent=2)

    print("\n✅ Pronto!")
    print(f"- cartoons_trakt.csv  (linhas: {len(big)})")
    print(f"- cartoons_trakt.json (linhas: {len(big)})")


if __name__ == "__main__":
    main()