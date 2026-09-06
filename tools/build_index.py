"""資料（Markdown）をチャンク化し、ローカル埋め込みで索引を作る。

埋め込みは Ollama の nomic-embed-text。従量APIは使わない。
索引は numpy の .npz 1本。ベクトルDBを別途立てない（PoCの範囲を広げないため）。
"""
import glob, json, os, re
import numpy as np
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CORPUS = os.path.join(DATA, "corpus")
INDEX  = os.path.join(DATA, "index.npz")
EMB_MODEL = "nomic-embed-text"
CHUNK, OVERLAP = 900, 350   # 表が分断されて答えが読めなくなるため大きく取る

def embed(texts):
    out = []
    for t in texts:
        req = urllib.request.Request(
            "http://localhost:11434/api/embeddings",
            data=json.dumps({"model": EMB_MODEL, "prompt": t}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out.append(json.loads(r.read())["embedding"])
    v = np.array(out, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

def chunks_of(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    # 見出し(## )で区切って「ページ」とみなす。出典に節番号を出せるようにするため。
    pages = re.split(r"\n(?=## )", txt)
    src = os.path.basename(path)
    for pno, page in enumerate(pages, 1):
        body = re.sub(r"[ \t]+", " ", page).strip()
        if not body: continue
        i = 0
        while i < len(body):
            piece = body[i:i + CHUNK].strip()
            if len(piece) > 40:
                yield {"text": piece, "source": src, "page": pno}
            i += CHUNK - OVERLAP

def main():
    recs = [c for p in sorted(glob.glob(os.path.join(CORPUS, "*.md"))) for c in chunks_of(p)]
    print(f"  チャンク {len(recs)} 件 / 出典 {len(set(r['source'] for r in recs))} ファイル")
    vecs = embed([r["text"] for r in recs])
    np.savez(INDEX, vecs=vecs, meta=np.array(json.dumps(recs, ensure_ascii=False)))
    print(f"作成: {INDEX}  次元 {vecs.shape[1]} / {os.path.getsize(INDEX)/1024:.0f} KB")

if __name__ == "__main__":
    main()
