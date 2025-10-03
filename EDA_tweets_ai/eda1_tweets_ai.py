import pandas as pd
import re
from collections import Counter
import ast
import sys, csv  # robust CSV reading
import matplotlib.pyplot as plt

# Allow very large CSV fields (avoids "buffer overflow")
csv.field_size_limit(sys.maxsize)

# --- Load your data (robust) ---
PATH = "filtered_tweets_ai.csv"
try:
    df = pd.read_csv(
        PATH,
        engine="python",
        on_bad_lines="skip",      # skip malformed rows
        encoding="utf-8",
        encoding_errors="replace" # replace bad bytes
    )
except Exception:
    # Fallback if quotes/escaping are broken
    df = pd.read_csv(
        PATH,
        engine="python",
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
        quoting=csv.QUOTE_NONE,   # treat quotes as ordinary chars
        escapechar="\\"
    )

df["tweet"] = df["tweet"].astype(str).fillna("")

# --- Stopwords (English + Twitter-ish) ---
STOPWORDS = {
    "a","an","the","and","or","but","if","then","than","so","because","as","of","at","by","for","with",
    "about","into","through","during","before","after","to","from","in","on","over","under","again",
    "further","is","are","was","were","be","been","being","do","does","did","doing","have","has","had",
    "having","i","me","my","we","our","you","your","he","him","his","she","her","it","its","they","them",
    "their","this","that","these","those","what","which","who","whom","where","when","why","how",
    "not","no","nor","only","own","same","too","very","can","will","just","should","could","would",
    # Twitter-ish
    "rt","via","amp","https","http","co","t","im","dont","doesnt","didnt","youre","ive","id","cant","wont",
}

# --- Helpers ---
URL_RE = re.compile(r'http\S+|www\.\S+', flags=re.IGNORECASE)
MENTION_RE = re.compile(r'@[a-z0-9_]+', flags=re.IGNORECASE)
HASHTAG_RE = re.compile(r'#[a-z0-9_]+', flags=re.IGNORECASE)
WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-']*", flags=re.IGNORECASE)  # keeps gpt-4

def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = text.replace("&amp;", " ")
    return text

# --- Counters ---
word_counter = Counter()
hashtag_counter = Counter()
mention_counter = Counter()

# --- From tweet text ---
for raw in df["tweet"]:
    tx = clean_text(raw)

    # mentions & hashtags
    mention_counter.update(m.lower() for m in MENTION_RE.findall(tx))
    hashtag_counter.update(h.lstrip("#").lower() for h in HASHTAG_RE.findall(tx))

    # words (excluding mentions/hashtags)
    tx_no_tags = MENTION_RE.sub(" ", HASHTAG_RE.sub(" ", tx))
    tokens = [w for w in WORD_RE.findall(tx_no_tags)
              if w not in STOPWORDS and not w.isdigit() and len(w) > 1]
    word_counter.update(tokens)

# --- Also parse a separate 'hashtags' column if present (merge both sources) ---
if "hashtags" in df.columns:
    for cell in df["hashtags"].dropna():
        s = str(cell).strip()
        tags = []
        if s.startswith("[") and s.endswith("]"):
            # looks like a Python list string -> parse safely
            try:
                lst = ast.literal_eval(s)
                tags = [str(x).lstrip("#").lower() for x in lst if isinstance(x, (str, int, float))]
            except Exception:
                pass
        if not tags:
            # fallback: extract like tokens from any string format
            tags = [t.lower().lstrip("#") for t in re.findall(r"[#]?([A-Za-z0-9_]+)", s)]
        hashtag_counter.update([t for t in tags if t])

# --- Top N results ---
TOP_N = 25
top_words = pd.DataFrame(word_counter.most_common(TOP_N), columns=["word", "count"])
top_hashtags = pd.DataFrame(hashtag_counter.most_common(TOP_N), columns=["hashtag", "count"])
top_mentions = pd.DataFrame(mention_counter.most_common(TOP_N), columns=["mention", "count"])

print("\nTop words (no stopwords):")
print(top_words)

print("\nTop hashtags:")
print(top_hashtags)

print("\nTop mentions:")
print(top_mentions)

# Optionally save to CSV
top_words.to_csv("freq_top_words.csv", index=False)
top_hashtags.to_csv("freq_top_hashtags.csv", index=False)
top_mentions.to_csv("freq_top_mentions.csv", index=False)

# --- Plots (saved as PNG + shown) ---

def plot_barh(df_in, label_col, count_col, title, filename):
    if df_in.empty:
        print(f"No data to plot: {title}")
        return
    data = df_in.copy().sort_values(count_col, ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(data[label_col], data[count_col])
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel(label_col.capitalize())
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()

plot_barh(top_words, "word", "count", "Top Words (No Stopwords)", "top_words.png")
plot_barh(top_hashtags, "hashtag", "count", "Top Hashtags", "top_hashtags.png")
plot_barh(top_mentions, "mention", "count", "Top Mentions", "top_mentions.png")
