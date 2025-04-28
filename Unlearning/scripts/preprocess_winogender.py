import pandas as pd

df = pd.read_csv("../data/wg.tsv", sep="\t")
print("Loaded columns:", df.columns.tolist())

records = {}
for _, row in df.iterrows():
    sid, sentence = row["sentid"], row["sentence"]
    parts = sid.replace(".txt", "").split(".")
    if len(parts) < 4:
        continue
    key = ".".join(parts[:-1])
    gender = parts[-1]
    if key not in records:
        records[key] = {}
    records[key][gender] = sentence

print("\nSample grouped keys with genders:")
for k, v in list(records.items())[:10]:
    print(f"{k} → {list(v.keys())}")

output = []
for key, variants in records.items():
    if all(g in variants for g in ("male", "female", "neutral")):
        output.append({
            "group_key": key,
            "male": variants["male"],
            "female": variants["female"],
            "neutral": variants["neutral"]
        })

out_df = pd.DataFrame(output)
out_df.to_csv("../data/wg_contrastive.tsv", sep="\t", index=False)
print(f"Saved contrastive triples: {len(out_df)} → ../data/wg_contrastive.tsv")
