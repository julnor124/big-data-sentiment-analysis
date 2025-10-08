# === Colab: labela HELA datasetet (topp-1 + sannolikhet) med progress bar ===
# !pip -q install transformers pandas torch tqdm --upgrade

import os, math, pandas as pd, torch
from tqdm.auto import tqdm
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# ---- KONFIG ----
INPUT = "AfterChatGPT.csv"   # <-- kör en fil i taget
TEXT_COL_HINT = None                  # sätt t.ex. "Tweet" eller "tweet" om du vill låsa; annars auto
MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
ENCODING, SEP = "utf-8", ","
BATCH_SIZE = 128                      # sänk till 64 om VRAM tar slut
MAX_LEN = 300                          # 96–128 räcker för tweets

TEXT_CANDIDATES = ["Tweet","tweet","text","Text","full_text","content","message","body"]

def pick_text_col(df: pd.DataFrame, hint: Optional[str]) -> str:
    if hint and hint in df.columns:
        return hint
    for c in TEXT_CANDIDATES:
        if c in df.columns: return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in TEXT_CANDIDATES:
        if c.lower() in lower_map: return lower_map[c.lower()]
    raise ValueError(f"Hittar ingen textkolumn. Tillgängliga kolumner: {list(df.columns)}")

# ---- Device / modell ----
device = 0 if torch.cuda.is_available() else -1
print("Device:", "CUDA" if device == 0 else "CPU")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else None
)
pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    function_to_apply="softmax",
    device=device
)

# ---- Läs och kör HELA datasetet (ingen sampling) ----
in_path = os.path.join("/content", INPUT) if not os.path.isabs(INPUT) else INPUT
if not os.path.exists(in_path):
    raise FileNotFoundError(f"Hittar inte filen: {in_path}")

df = pd.read_csv(in_path, encoding=ENCODING, sep=SEP)
text_col = pick_text_col(df, TEXT_COL_HINT)
texts = df[text_col].fillna("").astype(str).tolist()
n = len(texts)
print(f"Rader: {n}  |  Textkolumn: {text_col}")

# Progress bar i batchar
labels, probs = [], []
num_batches = math.ceil(n / BATCH_SIZE)

# Warmup (liten batch) för snabbare first-batch
_ = pipe(texts[:min(256, n)], batch_size=min(BATCH_SIZE, 64), truncation=True, max_length=MAX_LEN)

for i in tqdm(range(0, n, BATCH_SIZE), total=num_batches, desc="Labeling", unit="batch"):
    batch_texts = texts[i : i + BATCH_SIZE]
    outs = pipe(batch_texts, batch_size=BATCH_SIZE, truncation=True, max_length=MAX_LEN)
    for r in outs:
        r0 = r[0] if isinstance(r, list) else r  # topp-1
        labels.append(r0["label"])
        probs.append(float(r0.get("score", 0.0)))

# Spara
df_out = df.copy()
df_out["emotion_label"] = labels
df_out["emotion_prob"]  = probs

base = os.path.splitext(os.path.basename(in_path))[0]
out_path = f"/content/{base}.labeled.csv"
df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"✅ Sparade: {out_path}  (rader: {len(df_out)})")

from google.colab import files
files.download(out_path)
