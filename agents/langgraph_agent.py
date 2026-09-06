"""LangGraph 5ノードのエージェント（Plan → Retrieve → Query → Critique → Report）。

方針:
- 推論はローカルのみ（Ollama / qwen2.5:7b）。従量APIを使わない
- SQLは読み取り専用。DDL/DML は実行前に拒否する
- 自己検証で確信が低ければ Retrieve へ戻すが、ループは上限2回で必ず止める
"""
from __future__ import annotations
import json, os, re, sqlite3, sys, time
from typing import Annotated, Literal, TypedDict

import numpy as np
import urllib.request
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "shared"))
import core as _core

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
DB    = os.path.join(DATA, "sample.sqlite")
INDEX = os.path.join(DATA, "index.npz")
MODEL = os.environ.get("POC_MODEL", "qwen2.5:7b")
MAX_LOOP = 2

# 問答を全件記録するため、共有層のラッパで包む
# ノードごとに違うモデルを使う。実際に使うモデルは _core.NODE_MODELS が決める。
# 思考モデル（qwen3 系）は reasoning=False にしないと本文が空になる。core.chat_kwargs 参照。
llm      = _core.LoggedChat(
    lambda m: ChatOllama(model=m, temperature=0, **_core.chat_kwargs(m)), "text")
llm_json = _core.LoggedChat(
    lambda m: ChatOllama(model=m, temperature=0, format="json", **_core.chat_kwargs(m)), "json")

_z    = np.load(INDEX, allow_pickle=True)
_VECS = _z["vecs"]
_META = json.loads(str(_z["meta"]))

# スキーマ説明は3基盤で同一の材料を渡すため共有層から取る（比較の凍結契約）
SCHEMA = _core.SCHEMA

# 単語境界に頼らない。pragma_table_info のような関数形も止める。
FORBIDDEN = re.compile(
    r"(\b(drop|delete|update|insert|alter|create|replace|attach|vacuum|reindex|analyze)\b"
    r"|pragma|load_extension|readfile|writefile|edit\s*\()", re.I)
# 個人を特定しうる列は、SELECT に現れた時点で拒否する（質問文のゲートだけに頼らない）
PII_COLS = re.compile(r"\b(zip_code|birthday|latitude|longitude|ma_lead_id)\b", re.I)
# 回答本文に破壊的SQLが混ざっていないかの出力検査
OUT_DESTRUCTIVE = re.compile(
    r"(\b(drop\s+table|delete\s+from|truncate|update\s+\w+\s+set|insert\s+into|alter\s+table)\b"
    r"|(削除|消去|空に|初期化|破棄)(?:し|す|さ)[^。]{0,12}(ます|する|できます|してください))", re.I)

# 質問そのものを止める決定的ゲート。LLMの分類に任せない（2026-09-03 の評価で
# Q09/Q12 が route を誤り、破壊的SQLを回答本文に出力したため追加）。
ASK_PERSONAL = re.compile(
    r"(電話番号|メールアドレス|メール\s*アドレス|メアド|連絡先|氏名|個人を特定|名簿"
    r"|生年月日|誕生日|郵便番号|居住地|住所一覧"
    r"|住所[をのは]?\s*(一覧|列挙|出力|教え)"
    r"|(個々|個別|一人ひとり|顧客ごと|特定の顧客)[^。]{0,12}(緯度|経度|住所|位置|誕生日|生年月日|連絡先|出力)"
    r"|pid\s*[をのは]?\s*(一覧|列挙|出|教)|メール\s*[をのは]?\s*(一覧|列挙|出|教)"
    r"|e-?mail|mail\s*address|phone|\btel\b|postal\s*code|zip\s*code"
    r"|address(es)?\s*(list|一覧)|birth\s*?(day|date))", re.I)
# 決定的層は既知の語彙を網羅する。LLM判定(第2層)の揺れに依存させない。
# 2026-09-03 R2 で S02/S03/S04（空に/初期化/破棄）が第1層を素通りして
# LLM判定の当たり外れに委ねられていたため、語彙を広げた。
ASK_DESTRUCTIVE = re.compile(
    r"(削除|消して|消す|消去|全件消|空に|初期化|リセット|クリア|破棄|廃棄|抹消|消し去|白紙"
    r"|書き換え|上書き|更新して|挿入して"
    r"|全部\s*0|全て\s*0|すべて\s*0|ドロップ"
    r"|truncate|drop|delete|update|insert|alter|purge|empty|wipe|clear|reset"
    r"|remove|erase|flush|overwrite)", re.I)


class S(TypedDict):
    question: str
    route: str            # shiryo / hyou / both / reject
    reject_kind: str
    plan: str
    hits: list            # [{text, source, page, score}]
    sql: str
    rows: list
    sql_error: str
    score: int
    critique: str
    loop: int
    rewrite: str
    excerpt: str
    answer: str
    trace: Annotated[list, lambda a, b: a + b]


# ---------- ツール ----------
def _embed(text: str) -> np.ndarray:
    _t0 = time.time()
    req = urllib.request.Request(
        "http://localhost:11434/api/embeddings",
        data=json.dumps({"model": "nomic-embed-text", "prompt": text}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        v = np.array(json.loads(r.read())["embedding"], dtype=np.float32)
    # 埋め込みも「ローカルLLMとのやり取り」として記録する
    _core.record("embed", "embedding", text, f"{v.shape[0]}次元のベクトル",
                 time.time() - _t0, model=_core.EMB_MODEL)
    return v / (np.linalg.norm(v) + 1e-9)


def _terms(q: str) -> list:
    """質問から、そのまま資料に現れうる語を抜く（識別子・カタカナ・漢字語）。"""
    return [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}|[ァ-ヶー]{3,}|[一-龥]{2,}", q)
            if t.lower() not in ("ですか", "ください", "教えて")]


def search_docs(query: str, k: int = 3) -> list:
    """ベクトル類似度に語一致を混ぜたハイブリッド検索。

    ベクトルだけでは、比較表のように同じ語が繰り返される箇所で
    「主キー」のような要点が埋もれて上位に来ない（実測: gold が rank 7）。
    質問に含まれる語がチャンクに現れているかを加点する。
    """
    sims = _VECS @ _embed(query)
    ts = _terms(query)
    if ts:
        lex = np.array([
            sum(1 for t in ts if t.lower() in m["text"].lower()) / len(ts)
            for m in _META], dtype=np.float32)
        score = 0.7 * sims + 0.3 * lex
    else:
        score = sims
    idx = np.argsort(-score)[:k]
    return [{**_META[i], "score": round(float(score[i]), 3),
             "vec": round(float(sims[i]), 3)} for i in idx]


def run_sql(sql: str):
    """読み取り専用。SELECT / WITH 以外と危険語は実行しない。"""
    s = sql.strip().rstrip(";")
    if not re.match(r"^\s*(select|with)\b", s, re.I):
        return None, "拒否: SELECT / WITH 以外は実行しない"
    if FORBIDDEN.search(s):
        return None, "拒否: 書き込み・定義変更を含むSQLは実行しない"
    if ";" in s:
        return None, "拒否: 複文は実行しない"
    if PII_COLS.search(s):
        return None, "拒否: 個人を特定しうる列は取得しない"
    # SELECT * は列名検査を素通りして PII を返すため、列の明示を必須にする
    if re.search(r"select\s+(\w+\.)?\*", s, re.I):
        return None, "拒否: SELECT * は使わない。必要な列を明示すること"
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    # None を返すハンドラは中断しない。非ゼロを返して初めて abort する。
    deadline = time.time() + 5.0
    con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10000)
    try:
        cur = con.execute(s)
        cols = [d[0] for d in cur.description]
        # 結果側でも検査する（式や別名で PII 列を持ち出す経路を塞ぐ）
        if any(PII_COLS.search(c) for c in cols):
            return None, "拒否: 結果に個人を特定しうる列が含まれる"
        rows = [dict(zip(cols, r)) for r in cur.fetchmany(50)]
        return rows, ""
    except Exception as e:
        return None, f"SQLエラー: {e}"
    finally:
        con.close()


def _json(prompt: str, default: dict) -> dict:
    try:
        return json.loads(llm_json.invoke(prompt).content)
    except Exception:
        return default


# ---------- ノード ----------
def plan(s: S) -> dict:
    llm.set_node("plan"); llm_json.set_node("plan")
    q = s["question"]
    if ASK_PERSONAL.search(q):
        return {"route": "reject", "plan": "個人を特定する情報の要求",
                "trace": ["Plan: 安全ゲート(個人情報)→reject"]}
    if ASK_DESTRUCTIVE.search(q):
        return {"route": "reject", "plan": "データを変更・削除する要求",
                "trace": ["Plan: 安全ゲート(破壊的操作・正規表現)→reject"]}
    # 第2層: 「危険か」を聞かず、依頼が求める **行動** を分類して allowlist で通す。
    # boolean の unsafe 判定は婉曲表現に弱く、実測で
    # 「使わなくなったデータを片付けたい」「テーブルを畳んで」「set all flags to zero」が
    # すり抜けた。行動を取れば、言い回しが変わっても mutate は mutate になる。
    ALLOWED = {"count", "describe", "lookup"}
    g = _json(
        f"""このエージェントは読み取り専用のデータ照会です。依頼が求めている**行動**を1つ選んでください。
  count       … 件数・合計・平均・種類数など、集計して数える
  describe    … 仕様や定義の説明を読む
  lookup      … 条件に合う集計結果を引く
  mutate      … データの作成・変更・削除・初期化・再作成・片付け・整理・移動
  export_pii  … 個人を特定しうる情報（氏名・連絡先・住所・郵便番号・生年月日・
                 個々の顧客の位置情報）を出す

言い回しではなく、実行されたら何が起きるかで選んでください。
依頼: {q}
出力: {{"action": "count|describe|lookup|mutate|export_pii", "why": "1文"}}""",
        {"action": "mutate", "why": "分類に失敗したため拒否側へ倒す"})
    act = g.get("action", "mutate")
    if act not in ALLOWED:
        return {"route": "reject", "reject_kind": ("personal" if act == "export_pii" else "mutate"),
                "plan": f"行動分類 {act} は許可範囲外",
                "trace": [f"Plan: 安全ゲート(行動分類={act})→reject / {g.get('why','')}"]}
    d = _json(
        f"""あなたは分析エージェントの計画担当です。次の質問をJSONで分類してください。
質問: {q}

route は次から1つ:
  "shiryo" 仕様書・定義の説明で答えられる（表を触らない）
  "hyou"   データベースの集計で答えられる
  "both"   資料で定義を確認してから集計が要る
  "reject" 個人を特定する連絡先など、答えてはいけない

判断の目安:
  テーブル定義・主キー・列の意味・作成方法・元資料との差 → "shiryo"（資料に書いてある）
  件数・合計・平均・上位/下位・条件に合う顧客・**何種類あるか**・何件あるか → "hyou"（数えないと分からない）
  資料の定義を確認したうえで数える → "both"

出力: {{"route": "...", "plan": "手順を1文で"}}""",
        {"route": "both", "plan": "資料と表の両方を調べる"})
    route = d.get("route", "both")
    if route not in ("shiryo", "hyou", "both", "reject"):
        route = "both"
    return {"route": route, "plan": d.get("plan", ""),
            "trace": [f"Plan: route={route} / {d.get('plan','')}"]}


def retrieve(s: S) -> dict:
    llm.set_node("retrieve"); llm_json.set_node("retrieve")
    q = s["question"] if s["loop"] == 0 else \
        llm.invoke(f"次の質問を、社内資料を検索するための日本語の語句に言い換えて1行だけ出力してください。記号・絵文字・説明は付けない。\n{s['question']}").content.strip()
    hits = search_docs(q, k=3)
    return {"hits": hits, "rewrite": q,
            "trace": [f"Retrieve(loop={s['loop']}): q='{q[:40]}' top={[h['score'] for h in hits]}"]}


def query(s: S) -> dict:
    llm.set_node("query"); llm_json.set_node("query")
    ctx = "\n".join(f"[{h['source']} p{h['page']}] {h['text'][:900]}" for h in s["hits"])
    sql = llm.invoke(
        f"""SQLite の読み取りSQLを1文だけ出力してください。説明もコードフェンスも不要。SQL本文のみ。

# スキーマ
{SCHEMA}

# 参考（社内資料の抜粋）
{ctx}

# 規則
- SELECT または WITH で始める。書き込み・定義変更は禁止
- 日付は 'YYYY/MM/DD' の文字列。比較は文字列比較でよい
- 件数を聞かれたら COUNT(*) を使う
- 結果は50行以内に収める
- **質問に書かれていない絞り込み条件を勝手に追加しない**
- **数える対象がどちらの表のものかを先に決める。** 物件を数えるなら m_property だけ、
  顧客を数えるなら lead_plot_out だけを使う。両方の属性が要るときにだけ結合する
- **質問された指標に対応する列がスキーマに無い場合は、SQLを書かず `NO_COLUMN` とだけ出力する。**
  似た意味の別の列で代用しない

# 質問
{s['question']}""").content
    sql = re.sub(r"^```(?:sql)?|```$", "", sql.strip(), flags=re.M).strip()
    if sql.strip().upper().startswith("NO_COLUMN"):
        return {"sql": "", "rows": [], "sql_error": "該当する列がスキーマに存在しない",
                "trace": ["Query: NO_COLUMN（対応する列が無いため代用しない）"]}
    rows, err = run_sql(sql)
    tr = [f"Query: {sql[:90]}… → {'ERR:'+err if err else str(len(rows or []))+'行'}"]
    # 構文・列名の誤りは1回だけ自動修正する（拒否された場合は再試行しない）
    if err and not err.startswith("拒否"):
        fixed = llm.invoke(
            f"""次のSQLがエラーになりました。スキーマを見て、読み取りSQLを1文だけ出し直してください。
説明もコードフェンスも不要。SQL本文のみ。

# スキーマ
{SCHEMA}

# 失敗したSQL
{sql}

# エラー
{err}""").content
        fixed = re.sub(r"^```(?:sql)?|```$", "", fixed.strip(), flags=re.M).strip()
        rows2, err2 = run_sql(fixed)
        tr.append(f"Query(再試行): {fixed[:80]}… → {'ERR:'+err2 if err2 else str(len(rows2 or []))+'行'}")
        if not err2:
            sql, rows, err = fixed, rows2, ""
    # 文字列の完全一致で0件になった場合、表記ゆれを疑って LIKE で1回だけ試す
    if not err and rows and len(rows) == 1 and all(
            v in (0, None) for v in rows[0].values()) and re.search(r"=\s*'[^']+'", sql):
        loose = re.sub(r"=\s*('[^']+')", lambda m: f"LIKE {m.group(1)[:-1]}%'", sql, count=1)
        rows3, err3 = run_sql(loose)
        if not err3 and rows3 and any(v not in (0, None) for v in rows3[0].values()):
            tr.append(f"Query(表記ゆれ再試行): {loose[:80]}… → {rows3}")
            sql, rows = loose, rows3
    return {"sql": sql, "rows": rows or [], "sql_error": err, "trace": tr}


def critique(s: S) -> dict:
    llm.set_node("critique"); llm_json.set_node("critique")
    d = _json(
        f"""回答材料を自己採点してください。0〜5の整数。

質問: {s['question']}
資料の抜粋（これが実際に渡された全文）:
{chr(10).join(f"[{h['source']} p{h['page']} 類似度{h['score']}] {h['text']}" for h in s['hits']) or '(なし)'}
SQL: {s['sql'] or '(なし)'}
SQLの結果件数: {len(s['rows'])}
SQLの結果（先頭5件の実値）: {json.dumps(s['rows'][:5], ensure_ascii=False, default=str)}
SQLエラー: {s['sql_error'] or 'なし'}

score の目安: 5=質問に十分答えられる / 3=一部不足 / 0=材料が無い
重要: 資料の抜粋の中に質問への答えが**実際に書かれていない**なら 2 以下を付けること。
      「関連はあるが答えそのものが無い」は 2 とする。
出力: {{"score": 整数, "reason": "1文"}}""",
        {"score": 2, "reason": "判定不能（既定値。再検索させる)"})
    sc = int(d.get("score", 2)) if str(d.get("score", "")).lstrip("-").isdigit() else 2
    if s["sql_error"]:
        sc = min(sc, 2)
    return {"score": sc, "critique": d.get("reason", ""), "loop": s["loop"] + 1,
            "trace": [f"Critique: score={sc} loop={s['loop']+1} / {d.get('reason','')}"]}


def route_after_critique(s: S) -> Literal["retrieve", "report"]:
    if s["score"] < 3 and s["loop"] < MAX_LOOP:
        return "retrieve"
    return "report"


def report(s: S) -> dict:
    llm.set_node("report"); llm_json.set_node("report")
    cites = "\n".join(f"- {h['source']} p{h['page']}（類似度 {h['score']}）" for h in s["hits"])
    # 長い抜粋から該当箇所を拾えないため、先に「関係する記述だけ」を抜き出させる。
    # 抜粋をまとめて渡すと、7Bモデルは該当箇所を見つけられない（文脈の希釈）。
    # 実測: 3チャンク約2,700字を一括で渡すと「なし」を返すが、
    #       正解チャンク1本だけなら正しく抜き出せる。よって1本ずつ処理する。
    picks = []
    for h in s["hits"]:
        e = llm.invoke(
            f"""次の資料から、質問に答えている記述だけを原文のまま抜き出してください。
答えが書かれていなければ「なし」とだけ書いてください。要約・言い換えは禁止です。

質問: {s['question']}

資料 [{h['source']} p{h['page']}]:
{h['text']}""").content.strip()
        if e and e != "なし" and "なし" != e[:2]:
            picks.append(f"[{h['source']} p{h['page']}] {e}")
    excerpt = "\n".join(picks)
    n = len(s["rows"])
    zero_note = ""
    if s["sql"] and not s["sql_error"] and (n == 0 or (
            n == 1 and all(v in (0, None) for v in s["rows"][0].values()))):
        zero_note = ("\n★重要: SQLは成功したが結果が0件（または0）である。"
                     "これを事実として断定してはならない。"
                     "WHERE句の値の表記ゆれ（例『神奈川』と『神奈川県』）や、"
                     "列・テーブルの取り違えの可能性を必ず指摘し、"
                     "『条件に一致するデータが無い。条件の確認が必要』と書くこと。")
    use_note = ""
    if s["sql"] and not s["sql_error"] and n > 0:
        use_note = ("\n★重要: SQLが成功して結果が返っている。"
                    "この結果の数値を必ず結論に書くこと。"
                    "資料に記載が無いことを理由に『未確認』と書いてはならない。"
                    "また、資料に該当箇所が無いことを【不足】に書いてはならない。"
                    "SQLで答えが出ているなら【不足】は『なし』とすること。")
    ans = llm.invoke(
        f"""次の材料だけを使って日本語で回答してください。材料に無い数値を作らないでください。{use_note}{zero_note}

質問: {s['question']}
資料から抜き出した該当箇所:
{excerpt or '(なし)'}

資料の抜粋（全文）:
{chr(10).join(f"[{h['source']} p{h['page']}] {h['text'][:900]}" for h in s['hits'])}
実行したSQL: {s['sql'] or '(なし)'}
SQLの結果（先頭10件）: {json.dumps(s['rows'][:10], ensure_ascii=False, default=str)}
SQLエラー: {s['sql_error'] or 'なし'}
自己採点: {s['score']} / {s['critique']}

書式:
【結論】1〜3文
【根拠】使った出典（ファイル名とページ）と、使ったテーブル名
【不足】**質問に答えるうえで本当に足りなかったものだけ**を書く。
      答えが出ているなら「なし」と書く。
      「資料の抜粋に該当が無い」は、SQLや別の材料で答えが出ているなら不足ではない
※ 資料の抜粋に答えが書かれている場合は、必ずそれを引用して答えること。
   「明示されていません」と書いてよいのは、抜粋を読んでも本当に無いときだけ
※ 材料に無い数値は「未確認」と書くこと
※ 質問された指標が列として存在しない場合、別の列で代用しない。「その指標は保持していない」と書くこと
※ データを変更・削除するSQL文は、たとえ例示でも書かないこと""").content
    note = ""
    if OUT_DESTRUCTIVE.search(ans):
        ans = OUT_DESTRUCTIVE.sub("〈破壊的SQLのため削除〉", ans)
        note = "\n※ 回答に含まれていたデータ変更・削除のSQLは、出力検査により削除しました。"
    return {"excerpt": excerpt, "answer": f"{ans}{note}\n\n【出典】\n{cites}",
            "trace": ["Report: 生成" + ("（出力検査で破壊的SQLを除去）" if note else "")]}


def reject(s: S) -> dict:
    q = s["question"]
    if ASK_PERSONAL.search(q) or s.get("reject_kind") == "personal":
        msg = ("この質問には回答しません。ダミーデータは個人の連絡先を保持しておらず、"
               "個人を特定する情報の出力は用途外です。"
               "属性の集計（都道府県別の件数など）であれば対応できます。")
        why = "個人特定情報"
    else:
        msg = ("この操作は実行できません。本エージェントは読み取り専用で、"
               "データの削除・更新・挿入は行いません。"
               "また、破壊的なSQL文そのものも出力しません。"
               "件数の確認など、読み取りの範囲であれば対応できます。")
        why = "破壊的操作"
    return {"answer": msg, "score": 5, "trace": [f"Reject: {why}のため回答拒否"]}


def route_after_plan(s: S) -> Literal["retrieve", "query", "reject"]:
    if s["route"] == "reject": return "reject"
    if s["route"] == "hyou":   return "query"
    return "retrieve"


def route_after_retrieve(s: S) -> Literal["query", "critique"]:
    return "critique" if s["route"] == "shiryo" else "query"


def build():
    g = StateGraph(S)
    for n, f in [("plan", plan), ("retrieve", retrieve), ("query", query),
                 ("critique", critique), ("report", report), ("reject", reject)]:
        g.add_node(n, f)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", route_after_plan,
                            {"retrieve": "retrieve", "query": "query", "reject": "reject"})
    g.add_conditional_edges("retrieve", route_after_retrieve,
                            {"query": "query", "critique": "critique"})
    g.add_edge("query", "critique")
    g.add_conditional_edges("critique", route_after_critique,
                            {"retrieve": "retrieve", "report": "report"})
    g.add_edge("report", END)
    g.add_edge("reject", END)
    return g.compile()


def ask(question: str) -> dict:
    t0 = time.time()
    _core.new_request(question)
    out = build().invoke({"question": question, "loop": 0, "hits": [], "rows": [],
                          "sql": "", "sql_error": "", "score": 0, "rewrite": "", "excerpt": "", "reject_kind": "", "trace": []},
                         {"recursion_limit": 30})
    out["elapsed"] = round(time.time() - t0, 1)
    out["qa"] = list(_core.exchanges())
    return out



def stream_ask(question: str):
    """ノードが1つ終わるたびにイベントを流す（眺めるためのAPI）。

    LangGraph の stream(stream_mode="updates") をそのまま外へ出す。
    UI 側は「どのノードが動いたか」「何が確定したか」を逐次受け取れる。
    """
    t0 = time.time()
    _core.new_request(question)
    st = {"question": question, "loop": 0, "hits": [], "rows": [], "sql": "",
          "sql_error": "", "score": 0, "rewrite": "", "excerpt": "",
          "reject_kind": "", "trace": []}
    acc = dict(st)
    yield {"ev": "start", "q": question, "t": 0.0}
    sent = 0
    for chunk in build().stream(st, {"recursion_limit": 30}, stream_mode="updates"):
        for node, delta in chunk.items():
            if isinstance(delta, dict):
                for k, v in delta.items():
                    acc[k] = acc.get(k, []) + v if k == "trace" else v
            yield {"ev": "node", "node": node, "t": round(time.time() - t0, 2),
                   "route": acc.get("route", ""), "loop": acc.get("loop", 0),
                   "score": acc.get("score", 0),
                   "sql": acc.get("sql", ""), "sql_error": acc.get("sql_error", ""),
                   "rows": acc.get("rows", [])[:6],
                   "hits": [{"source": h["source"], "page": h["page"], "score": h["score"],
                             "text": h["text"][:160]} for h in acc.get("hits", [])],
                   "excerpt": (acc.get("excerpt") or "")[:400],
                   "trace": (acc.get("trace") or [])[-1:],
                   "qa": _core.exchanges()[sent:],
                   "answer": acc.get("answer", "")}
            sent = len(_core.exchanges())
    yield {"ev": "end", "t": round(time.time() - t0, 2), "qa": _core.exchanges()[sent:],
           "qa_total": len(_core.exchanges()),
           "answer": acc.get("answer", ""), "route": acc.get("route", "")}

if __name__ == "__main__":
    import sys
    r = ask(sys.argv[1] if len(sys.argv) > 1 else "反響のあった顧客は何人ですか")
    print("\n".join(r["trace"]))
    print("-" * 60)
    print(r["answer"])
    print(f"\n({r['elapsed']}秒 / score={r['score']} / loop={r['loop']})")
