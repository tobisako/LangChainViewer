#!/usr/bin/env python3
"""この Mac に入っている全モデルが Ollama で実際に動くかを1本ずつ確かめる。

一覧に出ることと、ロードして応答が返ることは別問題なので、実際に叩く。
大きいモデルを積み上げるとメモリを食い切るので、keep_alive=0 で毎回すぐ降ろす。
"""
import json, os, sys, time, urllib.error, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

PROMPT = "1+1は？　数字だけ答えて。"


def post(path, payload, timeout=900):
    req = urllib.request.Request(
        f"{core.OLLAMA}{path}", method="POST",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    with urllib.request.urlopen(f"{core.OLLAMA}/api/tags", timeout=10) as r:
        models = json.loads(r.read())["models"]
    models.sort(key=lambda m: -m.get("size", 0))

    print(f"  {'モデル':32}{'容量':>8}  {'秒':>6}  結果")
    print("  " + "-" * 78)
    ng = 0
    for m in models:
        name, gb = m["name"], m.get("size", 0) / 1e9
        t0 = time.time()
        try:
            if "embed" in name:  # 埋め込みモデルは生成APIでは動かない
                d = post("/api/embed", {"model": name, "input": "テスト", "keep_alive": 0})
                v = d.get("embeddings", [[]])[0]
                out = f"OK  {len(v)}次元のベクトルを返した"
            else:
                # 思考モデルは本文の前に思考を出すので、切っておかないと
                # 出力上限を思考で使い切って本文が空になる
                body = {"model": name, "prompt": PROMPT, "stream": False,
                        "keep_alive": 0, "options": {"num_predict": 24}}
                if core.thinks(name):
                    body["think"] = False
                d = post("/api/generate", body)
                txt = " ".join(d.get("response", "").split())[:40]
                note = "（思考オフ）" if core.thinks(name) else ""
                out = f"OK  「{txt}」{note}" if txt else "NG  空の応答"
                if not txt:
                    ng += 1
        except Exception as e:
            out, ng = f"NG  {type(e).__name__}: {str(e)[:44]}", ng + 1
        print(f"  {name:32}{gb:6.1f}GB  {time.time()-t0:6.1f}  {out}")
    print("  " + "-" * 78)
    print(f"  {len(models)}モデル中 {len(models)-ng} が Ollama で応答", end="")
    print(" ／ 全部動いた" if ng == 0 else f" ／ {ng}件 失敗")


if __name__ == "__main__":
    main()
