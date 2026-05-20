"""Data ingestion adapters.

Each module here exposes one or both of:

    fetch_games(season:int) -> pd.DataFrame
    fetch_playbyplay(game_id:str) -> pd.DataFrame

All adapters return a canonical schema documented in `schema.py`. The
cross-source ID join is in `id_mapping.py` (a shared concern).
"""
