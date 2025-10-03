import pandas as pd
import sys, csv, os, math

# --- Config ---
PATH_IN   = "clean_tweets_ai.csv"
PATH_OUT  = "clean_tweets_ai.downsampled.csv"

# Option A: fixed fraction to remove per day (e.g., 0.45 means keep 55%)
FRAC_REMOVE = 0.40

# Option B: aim for a target row count (set to an int to enable; set to None to ignore)
TARGET_ROWS = None  # e.g., 490_000

os.makedirs(os.path.dirname(PATH_OUT) or ".", exist_ok=True)

# --- Robust read ---
csv.field_size_limit(sys.maxsize)
df = pd.read_csv(
    PATH_IN,
    engine="python",
    on_bad_lines="skip",
    encoding="utf-8",
    encoding_errors="replace"
)

# Ensure valid dates; only downsample rows that have a date
df["date"] = pd.to_datetime(df["date"], errors="coerce")
mask_has_date = df["date"].notna()
df_with_date = df.loc[mask_has_date].copy()
df_no_date   = df.loc[~mask_has_date].copy()  # left untouched

# Day key
df_with_date["day"] = df_with_date["date"].dt.floor("D")

# Decide drop fraction
if TARGET_ROWS is not None:
    n = len(df_with_date)
    keep_frac = min(1.0, TARGET_ROWS / n) if n > 0 else 1.0
    drop_frac = max(0.0, 1.0 - keep_frac)
    print(f"Aiming for ~{TARGET_ROWS:,} rows: keep≈{keep_frac:.4f}, drop≈{drop_frac:.4f}")
else:
    drop_frac = FRAC_REMOVE
    keep_frac = 1.0 - drop_frac
    print(f"Fixed drop fraction per day: drop={drop_frac:.4f}, keep={keep_frac:.4f}")

def pick_to_drop(g):
    n_drop = int(math.floor(len(g) * drop_frac))
    if n_drop <= 0:
        return g.iloc[0:0]
    return g.sample(n=n_drop, random_state=42)

# Indices to drop (per day)
to_drop_idx = (
    df_with_date.groupby("day", group_keys=False)
                .apply(pick_to_drop)
                .index
)

# Drop and stitch back rows without dates
df_with_date = df_with_date.drop(index=to_drop_idx).drop(columns="day")
df_out = pd.concat([df_with_date, df_no_date], ignore_index=True)

# Save
df_out.to_csv(PATH_OUT, index=False, encoding="utf-8")

print(f"Input rows:  {len(df):,}")
print(f"Dropped:     {len(to_drop_idx):,}")
print(f"Output rows: {len(df_out):,}")
print(f"Saved to:    {PATH_OUT}")
