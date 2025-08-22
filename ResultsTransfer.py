# Title: Global Hydrogen Trade and Investment Modelling: Business Strategies for Adoption
# Author: William Cope, wtc32
# University: University of Cambridge - Department of Chemical Engineering and Biotechnology
# Degree: MPhil in Advanced Chemical Engineering
# Purpose: Minimal helpers to serialise experiment outputs into a run directory
# Date: 2025-08-22
#
# Notes:

# ============================ Imports & Types ============================= #
# results_io.py  (keep this file minimal)
from pathlib import Path
import pandas as pd
import numpy as np
import json
import datetime as dt
from typing import Any, Dict
# ======================== Lightweight Serialisation ======================= #
# Convert common objects to JSON-serialisable forms (shape-only for DataFrames)
def _jsonify(x: Any):
    if isinstance(x, (pd.DataFrame, pd.Series)):  # return minimal metadata rather than full payload
        return {"__type__": "dataframe", "shape": list(x.shape)}
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):  # lists serialise cleanly; dtype info not required here
        return x.tolist()
    if isinstance(x, set):  # sets are not JSON-native; convert to list
        return list(x)
    return x
# ============================= Save Entrypoint ============================ #
def save_all(run_id: str, out_dir: str | Path, **artifacts) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # ensure base output directory exists
    run_dir = out_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)  # one folder per run for clean separation

    manifest: Dict[str, Dict[str, str]] = {
        "_meta": {"created_utc": dt.datetime.utcnow().isoformat() + "Z", "run_id": run_id},
        "files": {},
    }

    for name, obj in artifacts.items():
        if isinstance(obj, pd.DataFrame):
            fp = run_dir / f"{name}.parquet"
            obj.to_parquet(fp, index=False)
            manifest["files"][name] = {"type": "dataframe", "path": fp.name}
        elif isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
            fp = run_dir / f"{name}.json"
            with fp.open("w", encoding="utf-8") as f:
                json.dump(obj, f, default=_jsonify, ensure_ascii=False, indent=2)
            manifest["files"][name] = {"type": "json", "path": fp.name}
        elif isinstance(obj, np.ndarray):
            fp = run_dir / f"{name}.json"
            with fp.open("w", encoding="utf-8") as f:
                json.dump(obj.tolist(), f, ensure_ascii=False, indent=2)
            manifest["files"][name] = {"type": "json", "path": fp.name}
        else:
            fp = run_dir / f"{name}.json"
            with fp.open("w", encoding="utf-8") as f:
                json.dump(_jsonify(obj), f, ensure_ascii=False, indent=2)
            manifest["files"][name] = {"type": "json", "path": fp.name}

    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[results_io] Saved artifacts to: {run_dir}")
    return run_dir
