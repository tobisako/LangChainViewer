"""CrewAI 版。共有層（shared/core.py）を使い、オーケストレーションだけ CrewAI で書く。

比較の契約: データ・索引・検索・SQL実行器・安全の第1層・許可行動の仕様は共有。
CrewAI に委ねるのは、役割の切り方（Agent/Task/Crew）とプロンプト。
"""
from __future__ import annotations
import json, os, re, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared"))
import core

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

LLM_CFG = LLM(model=f"ollama/{core.MODEL}", base_url=core.OLLAMA, temperature=core.TEMPERATURE)
_STATE: dict = {}


@tool("search_docs")
def t_search(query: str) -> str:
    """社内資料を検索し、出典付きの抜粋を返す。引数は検索したい語句。"""
    hits = core.search_docs(query)
    _STATE.setdefault("hits", []).extend(hits)
    _STATE["trace"].append(f"search_docs('{query[:34]}') → {[h['score'] for h in hits]}")
    return "\n".join(f"[{h['source']} p{h['page']}] {h['text']}" for h in hits)


@tool("run_sql")
def t_sql(sql: str) -> str:
    """SQLite に読み取りSQLを1文だけ発行して結果を返す。書き込みは実行されない。"""
    rows, err = core.run_sql(sql)
    _STATE["sql"] = sql
    _STATE["sql_error"] = err
    _STATE["rows"] = rows or []
    _STATE["trace"].append(f"run_sql: {sql[:70]}… → {'ERR:'+err if err else str(len(rows or []))+'行'}")
    if err:
        return f"エラー: {err}"
    return json.dumps(rows[:20], ensure_ascii=False, default=str)


def _classify(q: str) -> str:
    """第2層。許可行動の分類（仕様は共有、聞き方は CrewAI 側の裁量）。"""
    out = LLM_CFG.call([{"role": "user", "content":
        f"{core.ACTION_SPEC}\n\n依頼: {q}\n\n"
        "行動名だけを1語で出力してください。説明は不要です。"}])
    m = re.search(r"count|describe|lookup|mutate|export_pii", str(out), re.I)
    return m.group(0).lower() if m else "mutate"   # 判定不能なら拒否側へ倒す


def ask(question: str) -> dict:
    t0 = time.time()
    _STATE.clear()
    _STATE.update({"hits": [], "rows": [], "sql": "", "sql_error": "", "trace": []})

    # 第1層（共有・決定的）
    if core.ASK_PERSONAL.search(question):
        _STATE["trace"].append("Gate1(個人情報・正規表現) → reject")
        return _done(question, "reject", core.REJECT_PII, t0)
    if core.ASK_DESTRUCTIVE.search(question):
        _STATE["trace"].append("Gate1(破壊的操作・正規表現) → reject")
        return _done(question, "reject", core.REJECT_MUTATE, t0)
    act = _classify(question)
    _STATE["trace"].append(f"Gate2(行動分類={act})")
    if act not in core.ALLOWED_ACTIONS:
        return _done(question, "reject",
                     core.REJECT_PII if act == "export_pii" else core.REJECT_MUTATE, t0)

    analyst = Agent(
        role="データ照会の担当者",
        goal="社内資料とダミーデータベースを使い、根拠を示して質問に答える",
        backstory="不動産の顧客データを扱ってきた分析担当。読み取り専用の権限しか持たない。",
        tools=[t_search, t_sql], llm=LLM_CFG, verbose=False, max_iter=core.MAX_LOOP + 2,
        allow_delegation=False)
    reporter = Agent(
        role="報告の担当者",
        goal="調べた材料だけを使い、結論・根拠・不足を書き分ける",
        backstory="材料に無い数値を書かないことを徹底する報告担当。",
        llm=LLM_CFG, verbose=False, allow_delegation=False)

    t1 = Task(
        description=(f"次の質問に答えるための材料を集めてください。\n質問: {question}\n\n"
                     f"# スキーマ\n{core.SCHEMA}\n\n"
                     "規則:\n"
                     "- 仕様や定義を問われたら search_docs を使う\n"
                     "- 件数・合計・平均を問われたら run_sql を使う。SELECT または WITH で始める\n"
                     "- 質問に書かれていない絞り込み条件を勝手に追加しない\n"
                     "- 物件を数えるなら m_property、顧客を数えるなら lead_plot_out を使う\n"
                     "- 該当する列がスキーマに無ければ、代用せず『該当する列が無い』と書く"),
        expected_output="使った資料の抜粋、実行したSQL、その結果", agent=analyst)
    t2 = Task(
        description=("集めた材料だけを使って日本語で回答してください。\n"
                     f"質問: {question}\n\n書式:\n"
                     "【結論】1〜3文\n【根拠】使った出典と表\n【不足】無ければ「なし」\n"
                     "※ 材料に無い数値は「未確認」と書く\n"
                     "※ SQLが成功して結果が返っていれば、その数値を必ず結論に書く\n"
                     "※ データを変更・削除するSQLは、例示でも書かない"),
        expected_output="上の書式の回答", agent=reporter, context=[t1])

    crew = Crew(agents=[analyst, reporter], tasks=[t1, t2],
                process=Process.sequential, verbose=False)
    try:
        out = str(crew.kickoff())
    except Exception as e:
        out = f"【結論】処理に失敗しました。\n【不足】{type(e).__name__}: {e}"
        _STATE["trace"].append(f"crew error: {type(e).__name__}")
    ans, note = core.scan_output(out)
    cites = "\n".join(f"- {h['source']} p{h['page']}（score {h['score']}）" for h in _STATE["hits"])
    return _done(question, "hyou" if _STATE.get("sql") else "shiryo",
                 f"{ans}{note}" + (f"\n\n【出典】\n{cites}" if cites else ""), t0)


def _done(q, route, answer, t0):
    return {"question": q, "route": route, "answer": answer,
            "sql": _STATE.get("sql", ""), "sql_error": _STATE.get("sql_error", ""),
            "rows": _STATE.get("rows", []), "hits": _STATE.get("hits", []),
            "loop": 0, "score": 5, "excerpt": "",
            "trace": _STATE.get("trace", []), "elapsed": round(time.time() - t0, 1)}


if __name__ == "__main__":
    r = ask(sys.argv[1] if len(sys.argv) > 1 else "兵庫県の顧客は何件ありますか")
    print("\n".join(r["trace"])); print("-" * 60); print(r["answer"])
    print(f"\n({r['elapsed']}秒 / route={r['route']})")
