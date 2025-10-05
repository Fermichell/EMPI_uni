import argparse, json, regex, collections, pandas as pd, pathlib
def load_docs(path):
    items=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                o=json.loads(line)
                items.append((o.get("title",""), o.get("textBody","")))
    return items

def normalize(s):
    s=s.lower()
    s=regex.sub(r"[\p{P}\p{S}\d]+"," ",s)
    s=regex.sub(r"\s+"," ",s).strip()
    return s

def load_stop(path):
    S=set()
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            w=line.strip().lower()
            if w: S.add(w)
    return S

ap=argparse.ArgumentParser()
ap.add_argument("--json", required=True)
ap.add_argument("--stop", required=True)
ap.add_argument("--top", type=int, default=50)
a=ap.parse_args()

docs=load_docs(a.json)
stop=load_stop(a.stop)
cnt=collections.Counter()
for t,b in docs:
    words=[w for w in normalize(t+" "+b).split() if w not in stop]
    cnt.update(words)

pathlib.Path("outputs").mkdir(exist_ok=True, parents=True)
pd.DataFrame(cnt.most_common(), columns=["word","freq"]).to_csv("outputs/word_frequencies.csv", index=False, encoding="utf-8")
top=[w for w,_ in cnt.most_common(a.top)]
with open("outputs/top_words.txt","w",encoding="utf-8") as f:
    for w in top: f.write(w+"\n")
print("Saved outputs/word_frequencies.csv and outputs/top_words.txt")