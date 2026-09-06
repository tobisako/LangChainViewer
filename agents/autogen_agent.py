"""AutoGen 版。共有層（shared/core.py）を使い、オーケストレーションだけ AutoGen で書く。

比較の契約: データ・索引・検索・SQL実行器・安全の第1層・許可行動の仕様は共有。
AutoGen に委ねるのは、会話の組み方（Agent／Team／終了条件）とプロンプト。
"""
from __future__ import annotations
import asyncio, json, os, re, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared"))
import core

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_ext.models.ollama import OllamaChatCompletionClient

_STATE: dict = {}


def search_docs(query: str) -> str:
    """社内資料を検索し、出典付きの抜粋を返す。引数は検索したい語句。"""
    hits = core.search_docs(query)
    _STATE.setdefault("hits", []).extend(hits)
    _STATE["trace"].append(f"search_docs('{query[:34]}') → {[h['score'] for h in hits]}")
    return "\n".join(f"[{h['source']} p{h['page']}] {h['text']}" for h in hits)


def run_sql(sql: str) -> str:
    """SQLite に読み取りSQLを1文だけ発行して結果を返す。書き込みは実行されない。"""
    rows, err = core.run_sql(sql)
    _STATE["sql"] = sql
    _STATE["sql_error"] = err
    _STATE["rows"] = rows or []
    _STATE["trace"].append(f"run_sql: {sql[:70]}… → {'ERR:'+err if err else str(len(rows or []))+'行'}")
    return f"エラー: {err}" if err else json.dumps(rows[:20], ensure_ascii=False, default=str)


def _client(**kw):
    return OllamaChatCompletionClient(model=core.MODEL, host=core.OLLAMA,
                                      options={"temperature": core.TEMPERATURE}, **kw)


async def _classify(q: str) -> str:
    """第2層。許可行動の分類（仕様は共有、聞き方は AutoGen 側の裁量）。"""
    c = _client()
    try:
        from autogen_core.models import UserMessage
        r = await c.create([UserMessage(
            content=f"{core.ACTION_SPEC}\n\n依頼: {q}\n\n行動名だけを1語で出力してください。",
            source="user")])
        m = re.search(r"count|describe|lookup|mutate|export_pii", str(r.content), re.I)
        return m.group(0).lower() if m else "mutate"
    except Exception:
        return "mutate"          # 判定不能なら拒否側へ倒す
    finally:
        await c.close()


async def _run(question: str) -> str:
    client = _client()
    analyst = AssistantAgent(
        name="analyst",
        model_client=client,
        tools=[search_docs, run_sql],
        system_message=(
            "あなたは読み取り専用のデータ照会担当です。\n"
            f"# スキーマ\n{core.SCHEMA}\n\n"
            "規則:\n"
            "- 仕様や定義を問われたら search_docs を使います\n"
            "- 件数・合計・平均を問われたら run_sql を使います。SELECT または WITH で始めます\n"
            "- 質問に書かれていない絞り込み条件を勝手に追加しません\n"
            "- 物件を数えるなら m_property、顧客を数えるなら lead_plot_out を使います\n"
            "- 該当する列がスキーマに無ければ、代用せず『該当する列が無い』と書きます\n"
            "材料が揃ったら、集めた内容をそのまま reporter に渡してください。"))
    reporter = AssistantAgent(
        name="reporter",
        model_client=client,
        system_message=(
            "あなたは報告担当です。analyst が集めた材料だけを使って日本語で回答します。\n"
            "書式:\n【結論】1〜3文\n【根拠】使った出典と表\n【不足】無ければ「なし」\n"
            "※ 材料に無い数値は「未確認」と書きます\n"
            "※ SQLが成功して結果が返っていれば、その数値を必ず結論に書きます\n"
            "※ データを変更・削除するSQLは、例示でも書きません\n"
            "回答の最後に TERMINATE と書いてください。"))
    # 終了条件を必ず置く（置かないと会話が止まらない）
    term = TextMentionTermination("TERMINATE") | MaxMessageTermination(core.MAX_LOOP * 3)
    team = RoundRobinGroupChat([analyst, reporter], termination_condition=term)
    last = ""
    try:
        res = await team.run(task=question)
        for m in res.messages:
            t = getattr(m, "content", "")
            if isinstance(t, str) and "【結論】" in t:
                last = t
        if not last and res.messages:
            last = str(getattr(res.messages[-1], "content", ""))
        _STATE["trace"].append(f"Team: メッセージ {len(res.messages)} 件で停止")
    except Exception as e:
        last = f"【結論】処理に失敗しました。\n【不足】{type(e).__name__}: {e}"
        _STATE["trace"].append(f"team error: {type(e).__name__}")
    finally:
        await client.close()
    return last.replace("TERMINATE", "").strip()


def ask(question: str) -> dict:
    t0 = time.time()
    _STATE.clear()
    _STATE.update({"hits": [], "rows": [], "sql": "", "sql_error": "", "trace": []})
    if core.ASK_PERSONAL.search(question):
        _STATE["trace"].append("Gate1(個人情報・正規表現) → reject")
        return _done(question, "reject", core.REJECT_PII, t0)
    if core.ASK_DESTRUCTIVE.search(question):
        _STATE["trace"].append("Gate1(破壊的操作・正規表現) → reject")
        return _done(question, "reject", core.REJECT_MUTATE, t0)
    act = asyncio.run(_classify(question))
    _STATE["trace"].append(f"Gate2(行動分類={act})")
    if act not in core.ALLOWED_ACTIONS:
        return _done(question, "reject",
                     core.REJECT_PII if act == "export_pii" else core.REJECT_MUTATE, t0)
    out = asyncio.run(_run(question))
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
