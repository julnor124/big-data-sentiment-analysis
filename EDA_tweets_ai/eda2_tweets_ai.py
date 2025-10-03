import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, csv

# --- Robust inläsning (tolerant mot konstiga rader/tecken) ---
csv.field_size_limit(sys.maxsize)
PATH = "filtered_tweets_ai.csv"
try:
    df = pd.read_csv(
        PATH,
        engine="python",
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace"
    )
except Exception:
    df = pd.read_csv(
        PATH,
        engine="python",
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
        quoting=csv.QUOTE_NONE,
        escapechar="\\"
    )

# Säkerställ strängar
df["tweet"] = df["tweet"].astype(str).fillna("")

# --- Längdberäkningar ---
df["char_count"] = df["tweet"].str.len()
df["word_count"] = df["tweet"].str.split().str.len()

# --- Snabb sammanfattning / indikatorer för brus ---
short3 = (df["word_count"] <= 3).sum()
short_chars = (df["char_count"] <= 15).sum()
total = len(df)

print("=== Summary ===")
print(df[["char_count", "word_count"]].describe())
print(f"\nTweets ≤3 ord: {short3} ({short3/total*100:.2f}%)")
print(f"Tweets ≤15 tecken: {short_chars} ({short_chars/total*100:.2f}%)")

# Percentiler för att sätta rimliga x-axlar
p99_chars = np.percentile(df["char_count"], 99)
p99_words = np.percentile(df["word_count"], 99)

# --- Plot 1: Histogram tecken (full skala, log-y för att se svans) ---
plt.figure(figsize=(10, 5))
plt.hist(df["char_count"], bins=100)
plt.yscale("log")  # avslöjar lågfrekventa outliers
plt.title("Distribution of Tweet Length (Characters) - Full Range (log scale)")
plt.xlabel("Characters per Tweet")
plt.ylabel("Frequency (log)")
# Markera median/medel
plt.axvline(df["char_count"].median(), linestyle="--", label=f"Median={df['char_count'].median():.0f}")
plt.axvline(df["char_count"].mean(), linestyle=":", label=f"Mean={df['char_count'].mean():.1f}")
plt.legend()
plt.tight_layout()
plt.savefig("hist_char_full.png", dpi=150)
plt.show()

# --- Plot 2: Histogram tecken (fokusera på typiskt intervall) ---
# Sätt övre gräns till max(280, p99) för att täcka normal tweets + ev. längre
upper_chars = int(max(280, p99_chars))
subset_chars = df[df["char_count"] <= upper_chars]["char_count"]

plt.figure(figsize=(10, 5))
plt.hist(subset_chars, bins=60)
plt.title(f"Distribution of Tweet Length (Characters) ≤ {upper_chars}")
plt.xlabel("Characters per Tweet")
plt.ylabel("Frequency")
plt.axvline(subset_chars.median(), linestyle="--", label=f"Median={subset_chars.median():.0f}")
plt.axvline(subset_chars.mean(), linestyle=":", label=f"Mean={subset_chars.mean():.1f}")
plt.legend()
plt.tight_layout()
plt.savefig("hist_char_trimmed.png", dpi=150)
plt.show()

# --- Plot 3: Histogram ord (trim till 99:e percentilen) ---
upper_words = int(max(50, p99_words))  # visa upp till åtminstone 50 ord
subset_words = df[df["word_count"] <= upper_words]["word_count"]

plt.figure(figsize=(10, 5))
plt.hist(subset_words, bins=min(upper_words, 100))
plt.title(f"Distribution of Tweet Length (Words) ≤ {upper_words}")
plt.xlabel("Words per Tweet")
plt.ylabel("Frequency")
plt.axvline(subset_words.median(), linestyle="--", label=f"Median={subset_words.median():.0f}")
plt.axvline(subset_words.mean(), linestyle=":", label=f"Mean={subset_words.mean():.1f}")
plt.legend()
plt.tight_layout()
plt.savefig("hist_words_trimmed.png", dpi=150)
plt.show()

# --- Extra: skriv ut några exempel på väldigt korta tweets (kan vara brus) ---
print("\nExempel på mycket korta tweets (≤3 ord):")
print(df.loc[df["word_count"] <= 3, "tweet"].head(10).to_string(index=False))
