import pandas as pd
import re

# Läs in CSV-filen
df = pd.read_csv("../EDA_ChatGPT/ChatGPT_pre_clean.csv")

# Behåll endast kolumnerna 'date' och 'tweet' (ändra vid behov om kolumnnamnen skiljer sig)
df = df[['Date', 'Tweet']]

# Ta bort rader där datum eller tweet saknas
df = df.dropna(subset=['Date', 'Tweet'])

# Funktion för att rensa texten i tweets
def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    # Ta bort användarnamn (@username)
    text = re.sub(r'@\w+', '', text)
    # Ta bort # men behåll ordet
    text = re.sub(r'#', '', text)
    # Ta bort länkar (http, https, www)
    text = re.sub(r'http\S+|www\S+', '', text)
    # Ta bort överflödiga mellanslag
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Applicera rengöringsfunktionen
df['Tweet'] = df['Tweet'].apply(clean_tweet)

# ta bort dubletterna
df = df.drop_duplicates(subset=["Date", "Tweet"], keep="first").reset_index(drop=True)

# Visa några rader för kontroll
print(df.head())

# (valfritt) Spara det rensade datasetet
df.to_csv("chatgpt_cleaned_it2.csv", index=False)
