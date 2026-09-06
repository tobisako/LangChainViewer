#!/usr/bin/env python3
"""この Mac に入っている Ollama モデルの一覧を出す。

用途:
  - どのモデルが使えるか（＝Ollama で動くか）を確認する
  - ノードごとの割り当てと突き合わせる
"""
import json, os, sys, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core


def main():
    with urllib.request.urlopen(f"{core.OLLAMA}/api/tags", timeout=10) as r:
        models = json.loads(r.read()).get("models", [])
    models.sort(key=lambda m: -m.get("size", 0))
    used = {v: k for k, v in core.NODE_MODELS.items()}
    used[core.EMB_MODEL] = "embed"
    used[core.EMB_MODEL + ":latest"] = "embed"

    tot = sum(m.get("size", 0) for m in models)
    print(f"  {'モデル':30}{'容量':>9}  {'パラメータ':>10}  {'量子化':>8}  用途")
    print("  " + "-" * 82)
    for m in models:
        d = m.get("details", {}) or {}
        gb = m.get("size", 0) / 1_000_000_000
        role = used.get(m["name"], used.get(m["name"].split(":")[0], ""))
        print(f"  {m['name']:30}{gb:7.1f}GB  {d.get('parameter_size','-'):>10}  "
              f"{d.get('quantization_level','-'):>8}  {role}")
    print("  " + "-" * 82)
    print(f"  {len(models)} モデル / 合計 {tot/1_000_000_000:.1f}GB")
    print("\n  ノードごとの割り当て:")
    for k, v in core.NODE_MODELS.items():
        have = any(m["name"] == v for m in models)
        print(f"    {k:9} → {v:26} {'取得済み' if have else '未取得'}")
    print(f"    {'embed':9} → {core.EMB_MODEL}")


if __name__ == "__main__":
    main()
