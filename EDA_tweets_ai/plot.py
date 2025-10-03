import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("filtered_tweets_ai.csv")

# Make sure date is datetime and tweet is string
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["tweet"] = df["tweet"].astype(str)

# Ensure word/char counts exist
df["char_count"] = df["tweet"].str.len()
df["word_count"] = df["tweet"].str.split().str.len()

# --- 1. Tweets per month (time series) ---
tweets_per_month = df["date"].dt.to_period("M").value_counts().sort_index()

plt.figure(figsize=(12,5))
tweets_per_month.plot(kind="line", marker="o")
plt.title("Number of Tweets per Month")
plt.xlabel("Month")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=45)
plt.show()

# --- 2. Histogram: Tweet length (characters) ---
plt.figure(figsize=(10,5))
plt.hist(df["char_count"], bins=50, color="skyblue", edgecolor="black")
plt.title("Distribution of Tweet Length (Characters)")
plt.xlabel("Characters per Tweet")
plt.ylabel("Frequency")
plt.show()

# --- 3. Histogram: Tweet length (words) ---
plt.figure(figsize=(10,5))
plt.hist(df["word_count"], bins=50, color="lightgreen", edgecolor="black")
plt.title("Distribution of Tweet Length (Words)")
plt.xlabel("Words per Tweet")
plt.ylabel("Frequency")
plt.show()

# --- 4. Barplot: Retweets vs Originals (if column exists) ---
if "retweet" in df.columns:
    df["retweet"].value_counts().plot(kind="bar", color=["lightcoral", "lightblue"])
    plt.title("Retweets vs Original Tweets")
    plt.xlabel("Retweet")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.show()

# --- 5. Boxplot: Tweet length (characters) ---
plt.figure(figsize=(8,5))
plt.boxplot(df["char_count"], vert=False)
plt.title("Boxplot of Tweet Length (Characters)")
plt.xlabel("Characters per Tweet")
plt.show()
