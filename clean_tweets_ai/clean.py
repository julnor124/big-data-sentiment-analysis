import pandas as pd
import re
import sys, csv
import os

# -----------------------------
# CONFIG
# -----------------------------
PATH_IN  = "filtered_tweets_ai.csv"
PATH_OUT = "clean_tweets_ai.csv"

# Ensure output folder exists
os.makedirs(os.path.dirname(PATH_OUT) or ".", exist_ok=True)

# Stopwords (extend as needed)
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
# Normalize stopwords for apostrophe-less comparison (so "it's" → "its")
STOPWORDS_NORM = { re.sub(r"[’']", "", w) for w in STOPWORDS }

# -----------------------------
# Robust CSV load (read-only)
# -----------------------------
csv.field_size_limit(sys.maxsize)
try:
    df = pd.read_csv(
        PATH_IN,
        engine="python",
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace"
    )
except Exception:
    df = pd.read_csv(
        PATH_IN,
        engine="python",
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
        quoting=csv.QUOTE_NONE,
        escapechar="\\"
    )

# Work on a COPY so the original df stays untouched
df_clean = df.copy()

# Remove word count and char count left from the EDA
columns_to_drop = ["char_count", "word_count"]
df_clean = df_clean.drop(columns=columns_to_drop, errors="ignore")

# 2) Remove exact duplicates: same tweet AND same date
n_before = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=["tweet", "date"], keep="first")
print(f"Removed exact tweet+date duplicates: {n_before - len(df_clean)}")

# --- Cleaning helpers ---
URL_RE      = re.compile(r"http\S+|www\.\S+", flags=re.IGNORECASE)
# Remove mentions entirely, including optional trailing possessive (e.g., "@user's")
MENTION_RE  = re.compile(r"@[A-Za-z0-9_]+(?:’s|'s)?")
EMOJI_RE    = re.compile(
    "["                     
    "\U0001F1E0-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE
)
# Keep underscores & hyphens in word tokens (we'll remove hyphens in output),
# and ALSO capture basic punctuation as separate tokens.
TOKEN_RE    = re.compile(r"[a-z0-9][a-z0-9_\-']*|[.,!?:;]", flags=re.IGNORECASE)
WS_RE       = re.compile(r"\s+")

PUNCT_KEEP = {".", "!", "?", ";", ":"}

def clean_one(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" ", text)                   # remove links
    text = MENTION_RE.sub(" ", text)               # remove mentions (+ possessive)
    text = EMOJI_RE.sub(" ", text)                 # remove emojis/symbols
    # strip '#' but keep the hashtag word (e.g., "#OpenAI" -> "OpenAI")
    text = re.sub(r"#([A-Za-z0-9_'-]+)", r"\1", text)

    tokens = TOKEN_RE.findall(text)

    kept_tokens = []
    for t in tokens:
        if t in PUNCT_KEEP:
            # Keep punctuation as-is
            kept_tokens.append(t)
            continue

        # For stopword check: normalize by removing apostrophes and hyphens
        t_stop = t.replace("’", "'").replace("'", "").replace("-", "")
        if t_stop in STOPWORDS_NORM or t_stop.isdigit() or t_stop == "":
            continue

        # For output: remove hyphens only (keep apostrophes)
        t_out = t.replace("-", "")
        kept_tokens.append(t_out)

    # Attach punctuation to the previous token (no leading space before . ! ? ; :)
    out = []
    for tok in kept_tokens:
        if tok in PUNCT_KEEP:
            if out:
                out[-1] = out[-1] + tok
            else:
                out.append(tok)
        else:
            out.append(tok)

    out_text = " ".join(out)
    out_text = WS_RE.sub(" ", out_text).strip()    # collapse whitespace
    return out_text

# Overwrite 'tweet' with cleaned text (no extra column)
df_clean["tweet"] = df_clean["tweet"].astype(str).fillna("").map(clean_one)

# --- Save to a NEW CSV and report shape ---
df_clean.to_csv(PATH_OUT, index=False, encoding="utf-8")
print(f"Saved cleaned dataset to: {PATH_OUT}")
print(f"New CSV shape (rows, columns): {df_clean.shape}")
