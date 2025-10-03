import pandas as pd

# Load CSV into a DataFrame
df = pd.read_csv("data_files/tweets_ai.csv")

# Drop columns by name
columns_to_drop = ["id", "conversation_id", "created_at", "time", "timezone", 
                   "urls", "photos", "replies_count", "retweets_count", 
                   "likes_count", "cashtags", "retweet", "quote_url", 
                   "video", "thumbnail"]
df = df.drop(columns=columns_to_drop)

# Keep only English rows
df = df[df["language"] == "en"]

keywords = [
    # Core GPT terms
    "gpt", "gpt-2", "gpt-3", "gpt-3.5", "gpt-4", "gpt2", "gpt3", "gpt35", "gpt4",
    "chatgpt", "chat-gpt", "chat gpt", "chatgpt-2", "chatgpt-3", "chatgpt-3.5", "chatgpt-4",

    # Company / brand
    "openai", "open ai", "@openai",

    # OpenAI model families
    "davinci", "text-davinci", "davinci-003", "davinci-002",
    "curie", "babbage", "ada",

    # API / product references
    "openai api", "openai.com", "openaiapi",
    "playground.openai", "openai playground",

    # Other OpenAI models
    "instructgpt", "codex",

    # LLM-related terms (generic, broader net)
    "llm", "llms", "large language model", "large language models"
    
    # names of founders of OpenAI and leadership
    "sam altman", "altman", "@sama", "ilya sutskever", "@ilyasut", "greg brockman", "@gdb", "wojciech zaremba", "@woj_zaremba", "john schulman", "johnschulman2", "dario amodei", "@darioamodei", "vicki cheung", "@vmcheung", "pamela vagata"
]




# Keep rows where 'tweet' or 'hashtags' contains any keyword
mask = (
    df["tweet"].str.contains("|".join(keywords), case=False, na=False) |
    df["hashtags"].str.contains("|".join(keywords), case=False, na=False)
)

df = df[mask]

# Save the filtered DataFrame to a new CSV
df.to_csv("filtered_tweets_ai.csv", index=False)

