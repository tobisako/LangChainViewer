"""3基盤を同じ凍結セットで測るベンチ。

使い方:
  .venv-langgraph/bin/python run_bench.py langgraph holdout7
  .venv-crewai/bin/python    run_bench.py crewai    holdout7
  .venv-autogen/bin/python   run_bench.py autogen   safety_holdout4

各フレームワークは自分の venv から起動する（依存が同居できないため）。
評価セットと採点器は shared/ の同一物を使う。
"""
import csv, importlib, os, statistics, sys, time
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "shared"))
sys.path.insert(0, os.path.join(ROOT, "agents"))
sys.path.insert(0, ROOT)
import judge as J

MODULES = {"langgraph": "langgraph_agent", "crewai": "crewai_agent", "autogen": "autogen_agent"}


def main():
    fw, setname = sys.argv[1], sys.argv[2]
    mod = importlib.import_module(MODULES[fw])
    qs = yaml.safe_load(open(os.path.join(ROOT, "shared", "evals", f"eval_{setname}.yaml"),
                             encoding="utf-8"))
    # モデル構成（POC_PROFILE）を変えて測り直すので、既定以外は出力名に構成を混ぜる。
    # 既定 small のファイル名は変えない（既存のレポートが参照しているため）。
    import core
    tag = fw if core.PROFILE == "small" else f"{fw}-{core.PROFILE}"
    out = os.path.join(ROOT, "results", f"{tag}_{setname}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for q in qs:
        r = mod.ask(q["q"])
        r["rows_data"] = r["rows"]
        r["rows"] = len(r["rows"])
        ok, why = J.judge(q, r)
        rows.append({"fw": fw, "profile": core.PROFILE, "id": q["id"], "type": q["type"], "question": q["q"],
                     "judge": "OK" if ok else "NG", "why": why, "route": r["route"],
                     "sec": r["elapsed"], "sql": (r["sql"] or "").replace("\n", " ")[:160],
                     "rows": r["rows"], "answer": r["answer"].replace("\n", " ")[:300]})
        print(f"  {q['id']} [{q['type']:6}] {'OK' if ok else 'NG'} "
              f"{r['route']:6} {r['elapsed']:5.1f}s  {why[:52]}")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    ok = sum(1 for r in rows if r["judge"] == "OK")
    print(f"\n[{tag} / {setname}] 合格 {ok}/{len(rows)} / "
          f"平均 {statistics.mean(r['sec'] for r in rows):.1f}秒 → {out}")


if __name__ == "__main__":
    main()
