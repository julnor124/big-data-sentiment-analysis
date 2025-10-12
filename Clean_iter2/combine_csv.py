import pandas as pd

# Läs in dina två CSV-filer
df1 = pd.read_csv("chatgpt_cleaned_it2.csv")
df2 = pd.read_csv("generativeaiopinion_pre_clean.csv")

# Spara antal rader innan sammanslagning
rows_df1 = len(df1)
rows_df2 = len(df2)

# Hitta gemensamma kolumner
common_cols = list(set(df1.columns) & set(df2.columns))

# Behåll bara dessa kolumner i båda datamängderna
df1 = df1[common_cols]
df2 = df2[common_cols]

# Kombinera dem (radvis)
combined = pd.concat([df1, df2], ignore_index=True)

# Antal rader innan vi tar bort dubbletter
rows_before_dedup = len(combined)

# Ta bort eventuella dubbletter
combined = combined.drop_duplicates()

# Antal rader efter dubblettborttagning
rows_after_dedup = len(combined)

# Räkna hur många dubbletter som togs bort
duplicates_removed = rows_before_dedup - rows_after_dedup

# --- Ange kolumnordning explicit ---
desired_order = ["Date", "Tweet"]
combined = combined[desired_order]

# Spara till ny CSV-fil
combined.to_csv("postlaunch.csv", index=False)

# --- Utskrift ---
print("----- Sammanfattning -----")
print(f"chatgpt_cleaned.csv: {rows_df1} rader")
print(f"genaiop_cleaned.csv: {rows_df2} rader")
print(f"Totalt innan sammanslagning: {rows_df1 + rows_df2}")
print(f"Efter sammanslagning (före dubblettborttagning): {rows_before_dedup}")
print(f"Dubbletter borttagna: {duplicates_removed}")
print(f"Totalt efter sammanslagning: {rows_after_dedup}")
print(f"Kolumnordning i postlaunch.csv: {list(combined.columns)}")
print("---------------------------")

