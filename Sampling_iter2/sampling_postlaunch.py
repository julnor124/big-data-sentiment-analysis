#!/usr/bin/env python3
"""
Balanced Year × Emotion Sampler
===============================

Creates a sampled dataset (default 400 rows) spread evenly between:
  - emotion_label categories, and
  - years parsed from a date column.

Now with auto-detection of column names:
- Date column candidates: date, Date, created_at, timestamp
- Label column candidates: emotion_label, label, emotion

"""

import argparse
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def resolve_column(df, preferred, fallbacks, name_for_msg):
    """Return the first column that exists among [preferred] + fallbacks; else None."""
    candidates = [preferred] + [c for c in fallbacks if c != preferred]
    for c in candidates:
        if c in df.columns:
            if c != preferred:
                print(f"ℹ️ Using '{c}' as {name_for_msg} (auto-detected).")
            return c
    return None

def compute_targets(keys, total_target, availability=None, seed=42):
    """Evenly split total_target across keys, distributing remainder sensibly."""
    rng = np.random.default_rng(seed)
    n = len(keys)
    if n == 0:
        return {}
    base = total_target // n
    rem = total_target % n
    targets = {k: base for k in keys}
    if availability:
        order = sorted(keys, key=lambda k: availability.get(k, 0), reverse=True)
    else:
        order = list(keys)
        rng.shuffle(order)
    for k in order[:rem]:
        targets[k] += 1
    return targets

def balanced_sample(df, date_col="date", label_col="emotion_label",
                    sample_size=400, seed=42, verbose=True):
    """Balanced sampling over Year × Emotion with graceful fallback filling."""
    rng = np.random.default_rng(seed)

    # Parse year from date_col
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["Year"] = df[date_col].dt.year

    work = df.dropna(subset=["Year", label_col]).copy()
    work["Year"] = work["Year"].astype(int)

    years = sorted(work["Year"].unique().tolist())
    emotions = sorted(work[label_col].dropna().unique().tolist())

    if verbose:
        print("🌱 Balanced Year × Emotion Sampler")
        print("Timestamp:", datetime.now())
        print("📦 Rows after dropping missing year/label:", len(work))
        print("🗓️ Years:", years)
        print("🎭 Emotions:", emotions)

    if len(work) == 0 or len(years) == 0 or len(emotions) == 0:
        if verbose:
            print("❌ Not enough data to sample.")
        return work.head(0)

    # Availability
    year_avail = work.groupby("Year").size().to_dict()
    emo_avail  = work.groupby(label_col).size().to_dict()

    # Targets are caps, not guarantees
    year_targets = compute_targets(years, sample_size, availability=year_avail, seed=seed)
    emo_targets  = compute_targets(emotions, sample_size, availability=emo_avail,  seed=seed)

    if verbose:
        print("\n🎯 Target caps (ceilings):")
        print("   sum(year_targets):", sum(year_targets.values()))
        print("   sum(emo_targets): ", sum(emo_targets.values()))

    # Pre-shuffle indices within each (year, emotion) cell
    cell_indices = {}
    for (y, e), grp in work.groupby(["Year", label_col]):
        idx = grp.index.to_list()
        rng.shuffle(idx)
        cell_indices[(y, e)] = idx

    selected = []
    year_count = {y: 0 for y in years}
    emo_count  = {e: 0 for e in emotions}

    year_order = years.copy()
    emo_order  = emotions.copy()
    rng.shuffle(year_order)
    rng.shuffle(emo_order)

    def can_take(y, e):
        return (year_count[y] < year_targets[y] and
                emo_count[e]  < emo_targets[e]  and
                len(cell_indices.get((y, e), [])) > 0)

    # Phase 1: strict both caps
    progress = True
    while progress and len(selected) < sample_size:
        progress = False
        for y in year_order:
            for e in emo_order:
                if len(selected) >= sample_size:
                    break
                if can_take(y, e):
                    idx = cell_indices[(y, e)].pop()
                    selected.append(idx)
                    year_count[y] += 1
                    emo_count[e]  += 1
                    progress = True

    def rem_year(y): return max(0, year_targets[y] - year_count[y])
    def rem_emo(e):  return max(0, emo_targets[e]  - emo_count[e])

    # Phase 2: relax—prioritize cells with more remaining capacity
    cells_sorted = sorted(
        list(cell_indices.keys()),
        key=lambda ye: (rem_year(ye[0]) + rem_emo(ye[1])),
        reverse=True
    )
    i = 0
    while len(selected) < sample_size and i < len(cells_sorted):
        y, e = cells_sorted[i]
        if (year_count[y] < year_targets[y] or emo_count[e] < emo_targets[e]) and len(cell_indices[(y, e)]) > 0:
            idx = cell_indices[(y, e)].pop()
            selected.append(idx)
            year_count[y] += 1
            emo_count[e]  += 1
        else:
            i += 1

    # Phase 3: still short → fill from any remaining rows
    if len(selected) < sample_size:
        if verbose:
            print(f"\nℹ️ Caps unreachable. Filling remaining {sample_size - len(selected)} from any available rows.")
        pool = []
        for idxs in cell_indices.values():
            pool.extend(idxs)
        rng.shuffle(pool)
        need = sample_size - len(selected)
        selected.extend(pool[:need])

    sampled = work.loc[selected].copy()
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=seed)

    if verbose:
        print("\n✅ Sampling complete.")
        print(f"   Selected rows: {len(sampled)}")
        print("   Year counts:", sampled["Year"].value_counts().sort_index().to_dict())
        print("   Emotion counts:", sampled[label_col].value_counts().to_dict())

    return sampled.drop(columns=["Year"])

def main():
    parser = argparse.ArgumentParser(description="Create a balanced (Year × Emotion) sample from a labeled tweet dataset.")
    parser.add_argument("--input", required=True, help="Path to labeled CSV.")
    parser.add_argument("--output", required=True, help="Path to write the sampled CSV.")
    parser.add_argument("--sample-size", type=int, default=400, help="Total rows to sample. Default: 400.")
    parser.add_argument("--date-col", default="date", help="Date column name. Default: 'date'. Auto-detects common variants.")
    parser.add_argument("--label-col", default="emotion_label", help="Label column name. Default: 'emotion_label'. Auto-detects common variants.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    args = parser.parse_args()

    print("🌱 Balanced Year × Emotion Sampler")
    print("==================================")
    print("Timestamp:", datetime.now())
    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")
    print(f"Size:    {args.sample_size}")
    print(f"DateCol (requested): {args.date_col} | LabelCol (requested): {args.label_col}")
    print()

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"❌ Failed to read input CSV: {e}")
        sys.exit(1)

    # Auto-detect columns if needed
    date_col = resolve_column(
        df,
        preferred=args.date_col,
        fallbacks=["Date", "created_at", "timestamp"],
        name_for_msg="date column"
    )
    label_col = resolve_column(
        df,
        preferred=args.label_col,
        fallbacks=["label", "emotion"],
        name_for_msg="label column"
    )

    missing = []
    if date_col is None:
        missing.append(f"date column among {[args.date_col, 'Date', 'created_at', 'timestamp']}")
    if label_col is None:
        missing.append(f"label column among {[args.label_col, 'label', 'emotion']}")
    if missing:
        print("❌ Could not find required columns:", "; ".join(missing))
        print("   Available columns:", list(df.columns))
        sys.exit(1)

    sampled = balanced_sample(
        df,
        date_col=date_col,
        label_col=label_col,
        sample_size=args.sample_size,
        seed=args.seed,
        verbose=True
    )

    # --- Cluster by emotion, then by date (stable sort keeps sampling randomness within groups)
    sampled = sampled.sort_values(by=[label_col, date_col], kind="stable")

    if len(sampled) == 0:
        print("❌ No rows sampled. Exiting.")
        sys.exit(2)

    try:
        sampled.to_csv(args.output, index=False)
    except Exception as e:
        print(f"❌ Failed to write output CSV: {e}")
        sys.exit(1)

    print("\n📁 Saved sampled dataset →", args.output)
    print("Done.")

if __name__ == "__main__":
    main()

"""
Run this script with this command:
python -u Sampling_iter2/sampling_tweetsai.py \
  --input Sampling_iter2/tweets_ai_downsampled.labeled.csv \
  --output Sampling/tweets_ai_sampled_400.csv \
  --sample-size 400
"""