import argparse, json, regex, numpy as np, networkx as nx, pathlib
def load_docs(path):
    items=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                o=json.loads(line)
                items.append((o.get("title",""), o.get("textBody","")))
    return items

def split_sents(text):
    text=regex.sub(r"\s+"," ",text.strip())
    return [s for s in regex.split(r"[.!?]+\s*", text) if s]

def normalize(s):
    s=s.lower()
    s=regex.sub(r"[\p{P}\p{S}\d]+"," ",s)
    s=regex.sub(r"\s+"," ",s).strip()
    return s

ap=argparse.ArgumentParser()
ap.add_argument("--json", required=True)
ap.add_argument("--top_words", required=True)
a=ap.parse_args()

with open(a.top_words,"r",encoding="utf-8") as f:
    concepts=[w.strip().lower() for w in f if w.strip()]
idx={w:i for i,w in enumerate(concepts)}
M=len(concepts)
A=np.zeros((M,M), dtype=int)

docs=load_docs(a.json)
for t,b in docs:
    txt=f"{t}. {b}"
    for s in split_sents(txt):
        s=normalize(s)
        present=[w for w in concepts if w in s]
        for i in range(len(present)):
            for j in range(i+1,len(present)):
                p,q=idx[present[i]], idx[present[j]]
                A[p,q]+=1; A[q,p]+=1

np.fill_diagonal(A, 0)
pathlib.Path("outputs").mkdir(exist_ok=True, parents=True)
with open("outputs/concepts_adjacency.csv","w",encoding="utf-8") as f:
    f.write("Concept;"+";".join(concepts)+"\n")
    for i,w in enumerate(concepts):
        row=";".join(str(int(x)) for x in A[i])
        f.write(w+";"+row+"\n")

G=nx.Graph()
for w in concepts: G.add_node(w)
for i in range(M):
    for j in range(i+1,M):
        if A[i,j]>0: G.add_edge(concepts[i], concepts[j], weight=int(A[i,j]))
nx.write_gexf(G, "outputs/concepts_graph.gexf")
print("Saved outputs/concepts_adjacency.csv and outputs/concepts_graph.gexf")