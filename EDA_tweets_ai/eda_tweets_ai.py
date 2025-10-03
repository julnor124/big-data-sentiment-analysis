import pandas as pd

# Ladda in det filtrerade datasetet
df = pd.read_csv("tweets_ai.csv")

# Keep only English rows
df = df[df["language"] == "en"]

# Drop columns by name, kolumner vi inte vill ha
columns_to_drop = ["id", "conversation_id", "created_at", "time", "timezone", 
                   "urls", "photos", "replies_count", "retweets_count", 
                   "likes_count", "cashtags", "hashtags", "retweet", "quote_url", 
                   "video", "thumbnail", "language"]
df = df.drop(columns=columns_to_drop)

# --- 1. Datasetets form ---
print("Dataset shape (rows, columns):", df.shape)

# --- 2. Datatyper ---
print("\nDatatyper innan konvertering:")
print(df.dtypes)

# Se till att date är datetime och tweet är string
df["date"] = pd.to_datetime(df["date"], errors="coerce")   # konvertera datum
df["tweet"] = df["tweet"].astype(str)                      # säkerställ string

print("\nDatatyper efter konvertering:")
print(df.dtypes)

# --- 3. Saknade värden ---
print("\nAntal saknade värden:")
print(df[["date", "tweet"]].isna().sum())

# --- 4. Duplicerade tweets ---
duplicates = df["tweet"].duplicated().sum()
print(f"\nAntal duplicerade tweets (exakt samma text): {duplicates}")

# --- 5. Datumfördelning ---
print("\nAntal tweets per dag:")
print(df["date"].dt.date.value_counts().sort_index().head(10))  # de 10 första dagarna

print("\nAntal tweets per månad:")
print(df["date"].dt.to_period("M").value_counts().sort_index())

# --- 6. Tweetlängd ---
df["char_count"] = df["tweet"].str.len()         # antal tecken
df["word_count"] = df["tweet"].str.split().str.len()  # antal ord

print("\nTweetlängd (tecken):")
print(df["char_count"].describe())

print("\nTweetlängd (ord):")
print(df["word_count"].describe())

# Antal tweets med 3 ord eller färre
short_tweets_count = (df["word_count"] <= 5).sum()
print("Antal tweets med 3 ord eller färre:", short_tweets_count)

# Procentandel av hela datasetet
percentage = short_tweets_count / len(df) * 100
print(f"Andel av datasetet: {percentage:.2f}%")

# Save the filtered DataFrame to a new CSV
df.to_csv("filtered_tweets_ai.csv", index=False)
