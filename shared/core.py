"""3基盤で共有する層（比較の契約）。

比較評価の凍結契約に従い、ここに置いたものは全フレームワークで同一とする。
  - データ、索引、チャンク条件、埋め込み、検索方式、k
  - 推論モデルと temperature
  - SQL実行器と全てのガード
  - 第1層の正規表現（語を足さない）
  - 許可行動の仕様
各フレームワークが自由に決めてよいのは、オーケストレーションの切り方・プロンプト・
再検索の条件・レポートの書式であって、この層ではない。
"""
from __future__ import annotations
import json, os, re, sqlite3, time
import numpy as np
import urllib.request

# パスはこのファイルからの相対で解決する（別マシンへ持ち出せるようにするため）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
DB = os.path.join(DATA, "sample.sqlite")
INDEX = os.path.join(DATA, "index.npz")

MODEL = os.environ.get("POC_MODEL", "qwen2.5:7b")

# ノードごとに使うモデルを変える。
# 役割の重さに合わせてモデルを充て、どこに大きいモデルが要るかを実測できるようにする。
#
# プロファイル（環境変数 POC_PROFILE で切り替え。既定は small）:
#   small      … 全ノード小型。速いが賢さは落ちる（合計 8.8GB）
#   large      … 重いノードに大型を充てる。賢いが遅い（合計 75GB）
#   mixed      … 大型と無検閲を混在させる（合計 101GB）
#   uncensored … 生成ノードを無検閲モデルにする（合計 27GB）
#
# uncensored / mixed を用意しているのは趣味ではなく検証のため。
# 土台のモデル自身が持つ拒否（＝ベンダーの安全調整）に頼っていると、
# 「モデルを差し替えたら守れなくなる」設計なのかどうかが分からない。
# 無検閲モデルを生成ノードに置いて安全セットが通るなら、
# 守っているのはモデルではなくエージェント側の3層防御だと言える。
#
# 環境変数 POC_MODEL_<ノード名> で個別に上書きできる（プロファイルより優先）。
PROFILES = {
    "small": {
        "plan":     "qwen2.5:1.5b",        # 分類だけ。軽くてよい
        "retrieve": "llama3.2:1b",         # 検索語の言い換え
        "query":    "qwen2.5-coder:1.5b",  # SQL生成はコード寄り
        "critique": "gemma3:1b",           # 自己採点
        "report":   "qwen2.5:7b",          # 日本語の生成は重い
    },
    "large": {
        "plan":     "qwen2.5:1.5b",
        "retrieve": "llama3.2:1b",
        "query":    "qwen3:32b",           # 20GB。SQL生成を賢いモデルに
        "critique": "qwen3:32b",
        "report":   "qwen2.5:72b-instruct-q5_K_M",  # 54GB。日本語生成を最上位に
    },
    "mixed": {
        "plan":     "qwen2.5:1.5b",
        "retrieve": "llama3.2:1b",
        "query":    "qwen3:32b",             # 20GB
        "critique": "dolphin-mixtral:8x7b",  # 26GB・無検閲
        "report":   "qwen2.5:72b-instruct-q5_K_M",  # 54GB
    },
    "uncensored": {
        "plan":     "qwen2.5:1.5b",
        "retrieve": "llama3.2:1b",
        "query":    "qwen2.5-coder:1.5b",
        "critique": "gemma3:1b",
        "report":   "dolphin-mixtral:8x7b",  # 26GB・無検閲。生成が一切断らない状態
    },
}
PROFILE = os.environ.get("POC_PROFILE", "small")
if PROFILE not in PROFILES:
    raise SystemExit(f"POC_PROFILE={PROFILE} は未定義。使えるのは {list(PROFILES)}")
NODE_MODELS = {
    n: os.environ.get(f"POC_MODEL_{n.upper()}", m) for n, m in PROFILES[PROFILE].items()
}
# 思考（reasoning）を出すモデル。
# qwen3 系は本文の前に <think> の内容を出すため、出力上限が小さいと
# 思考だけで打ち切られて本文が空になる（実測: num_predict=24 で response が空）。
# このエージェントは思考の中身を使わないので、明示的に切る。
THINKING_MODELS = ("qwen3",)


def thinks(model: str) -> bool:
    """そのモデルが思考モードを持つか。"""
    return model.split(":")[0] in THINKING_MODELS


def chat_kwargs(model: str) -> dict:
    """ChatOllama に渡す、モデル固有の追加引数。"""
    return {"reasoning": False} if thinks(model) else {}


EMB_MODEL = os.environ.get("POC_EMB", "nomic-embed-text")
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TEMPERATURE = 0
TOP_K = 3
VEC_W, LEX_W = 0.7, 0.3
MAX_LOOP = 2

SCHEMA = """
lead_plot_agg (集計用) / lead_plot_out (出力用) — 30列。件数・構造とも同一。
  pid                    リードID
  request_date           反響日 'YYYY-MM-DD'
  first_visit_date       初回来場日
  address1               都道府県（※リード側の都道府県はこの列。prefecture 列は存在しない）
  address2               市区
  address3, address4_1   —／丁目
  zip_code, birthday     郵便番号 / 生年月日
  recognition_media      認知媒体（折込チラシ / Web検索 / 紹介 / SNS広告 / 現地看板 / 住宅情報誌 / テレビCM / その他）
  budget_price           予算（万円・数値）
  object_code            物件コード
  formal_object_name     物件名
  hope_building_area     希望建物面積（数値）
  register_date, subscription_date, contract_date, cancel_contract_date  各日付
    ※ これらの日付列は、欠損を空文字 '' で保持している（NULL ではない）。
  reaction_flg, visitor_flg, contractor_flg                 反響/来場/契約フラグ（0 or 1）
  contract_generations   契約世代（数値）
  visit_flg_current_month, contract_flg_current_month       当月フラグ（0 or 1）
  latitude, longitude    緯度経度（数値）
  budget_band            予算帯（文字列。例 '2500-3500'）
  duplicated_flag_ma, ma_lead_id                             重複フラグ / リードID

# 表の対応
#   リードの属性 … lead_plot_out / lead_plot_agg（都道府県は address1）
#   物件の属性   … m_property （都道府県は prefecture）
#   両者の関連   … object_code

m_property (物件マスタ) — prefecture / city はこちらの表にのみ存在する
  object_code, formal_object_name, prefecture, city,
  budget_min_man, budget_max_man, building_area_avg, latitude, longitude
"""

# ---------- 安全: 第1層（決定的）。語を足さない ----------
ASK_PERSONAL = re.compile(
    r"(電話番号|メールアドレス|メール\s*アドレス|メアド|連絡先|氏名|個人を特定|名簿"
    r"|生年月日|誕生日|郵便番号|居住地|住所一覧"
    r"|住所[をのは]?\s*(一覧|列挙|出力|教え)"
    r"|(個々|個別|一人ひとり|顧客ごと|特定の顧客)[^。]{0,12}(緯度|経度|住所|位置|誕生日|生年月日|連絡先|出力)"
    r"|pid\s*[をのは]?\s*(一覧|列挙|出|教)|メール\s*[をのは]?\s*(一覧|列挙|出|教)"
    r"|e-?mail|mail\s*address|phone|\btel\b|postal\s*code|zip\s*code"
    r"|address(es)?\s*(list|一覧)|birth\s*?(day|date))", re.I)
ASK_DESTRUCTIVE = re.compile(
    r"(削除|消して|消す|消去|全件消|空に|初期化|リセット|クリア|破棄|廃棄|抹消|消し去|白紙"
    r"|書き換え|上書き|更新して|挿入して"
    r"|全部\s*0|全て\s*0|すべて\s*0|ドロップ"
    r"|truncate|drop|delete|update|insert|alter|purge|empty|wipe|clear|reset"
    r"|remove|erase|flush|overwrite)", re.I)

# 許可する行動（第2層の仕様。聞き方は各フレームワークに委ねる）
ALLOWED_ACTIONS = {"count", "describe", "lookup"}
ACTION_SPEC = """このエージェントは読み取り専用のデータ照会です。依頼が求めている**行動**を1つ選んでください。
  count       … 件数・合計・平均・種類数など、集計して数える
  describe    … 仕様や定義の説明を読む
  lookup      … 条件に合う集計結果を引く
  mutate      … データの作成・変更・削除・初期化・再作成・片付け・整理・移動
  export_pii  … 個人を特定しうる情報（氏名・連絡先・住所・郵便番号・生年月日・
                 個々の顧客の位置情報）を出す

言い回しではなく、実行されたら何が起きるかで選んでください。"""

REJECT_MUTATE = ("この操作は実行できません。本エージェントは読み取り専用で、"
                 "データの削除・更新・挿入は行いません。また、破壊的なSQL文そのものも出力しません。"
                 "件数の確認など、読み取りの範囲であれば対応できます。")
REJECT_PII = ("この質問には回答しません。ダミーデータは個人の連絡先を保持しておらず、"
              "個人を特定する情報の出力は用途外です。"
              "属性の集計（都道府県別の件数など）であれば対応できます。")

# ---------- 安全: SQL実行器（全ガード） ----------
FORBIDDEN = re.compile(
    r"(\b(drop|delete|update|insert|alter|create|replace|attach|vacuum|reindex|analyze)\b"
    r"|pragma|load_extension|readfile|writefile|edit\s*\()", re.I)
PII_COLS = re.compile(r"\b(zip_code|birthday|latitude|longitude|ma_lead_id)\b", re.I)
OUT_DESTRUCTIVE = re.compile(
    r"(\b(drop\s+table|delete\s+from|truncate|update\s+\w+\s+set|insert\s+into|alter\s+table)\b"
    r"|(削除|消去|空に|初期化|破棄)(?:し|す|さ)[^。]{0,12}(ます|する|できます|してください))", re.I)


def run_sql(sql: str):
    """読み取り専用。SELECT / WITH 以外と危険語・SELECT *・PII列・複文は実行しない。"""
    s = sql.strip().rstrip(";")
    if not re.match(r"^\s*(select|with)\b", s, re.I):
        return None, "拒否: SELECT / WITH 以外は実行しない"
    if FORBIDDEN.search(s):
        return None, "拒否: 書き込み・定義変更を含むSQLは実行しない"
    if ";" in s:
        return None, "拒否: 複文は実行しない"
    if PII_COLS.search(s):
        return None, "拒否: 個人を特定しうる列は取得しない"
    if re.search(r"select\s+(\w+\.)?\*", s, re.I):
        return None, "拒否: SELECT * は使わない。必要な列を明示すること"
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    deadline = time.time() + 5.0
    con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10000)
    try:
        cur = con.execute(s)
        cols = [d[0] for d in cur.description]
        if any(PII_COLS.search(c) for c in cols):
            return None, "拒否: 結果に個人を特定しうる列が含まれる"
        return [dict(zip(cols, r)) for r in cur.fetchmany(50)], ""
    except Exception as e:
        return None, f"SQLエラー: {e}"
    finally:
        con.close()


def scan_output(text: str):
    """回答本文の決定的検査。破壊的な記述を除去して返す。"""
    if OUT_DESTRUCTIVE.search(text):
        return (OUT_DESTRUCTIVE.sub("〈破壊的SQLのため削除〉", text),
                "\n※ 回答に含まれていたデータ変更・削除のSQLは、出力検査により削除しました。")
    return text, ""


# ---------- 検索（ハイブリッド） ----------
_z = np.load(INDEX, allow_pickle=True)
_VECS = _z["vecs"]
_META = json.loads(str(_z["meta"]))


def embed(text: str) -> np.ndarray:
    _t0 = time.time()
    req = urllib.request.Request(
        f"{OLLAMA}/api/embeddings",
        data=json.dumps({"model": EMB_MODEL, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        v = np.array(json.loads(r.read())["embedding"], dtype=np.float32)
    try:
        record("embed", "embedding", text, f"{v.shape[0]}次元のベクトル", time.time() - _t0,
               model=EMB_MODEL)
    except Exception:
        pass
    return v / (np.linalg.norm(v) + 1e-9)


def _terms(q: str) -> list:
    return [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}|[ァ-ヶー]{3,}|[一-龥]{2,}", q)
            if t.lower() not in ("ですか", "ください", "教えて")]


def search_docs(query: str, k: int = TOP_K) -> list:
    sims = _VECS @ embed(query)
    ts = _terms(query)
    if ts:
        lex = np.array([sum(1 for t in ts if t.lower() in m["text"].lower()) / len(ts)
                        for m in _META], dtype=np.float32)
        score = VEC_W * sims + LEX_W * lex
    else:
        score = sims
    idx = np.argsort(-score)[:k]
    return [{**_META[i], "score": round(float(score[i]), 3),
             "vec": round(float(sims[i]), 3)} for i in idx]

# ---------- ローカルLLMとの問答を全件記録する ----------
# 目的は2つ。(1) 後から何を聞いて何が返ったかを追えること
#           (2) 画面に出して、動作中に人が読めること
LOG_DIR = os.path.join(ROOT, "logs")

# 複数のリクエストが同時に走るため（GUI と検証が並行するなど）、
# 問答の器はスレッドごとに分ける。共有すると互いに消し合う。
import threading
_TL = threading.local()


def exchanges() -> list:
    if not hasattr(_TL, "ex"):
        _TL.ex = []
    return _TL.ex


def new_request(question: str) -> str:
    """1回の質問の開始。以後の問答をこの ID にひも付ける。"""
    import uuid
    exchanges().clear()
    _TL.req = uuid.uuid4().hex[:12]
    _log({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "req": _TL.req,
          "kind": "question", "text": question})
    return _TL.req


def _log(rec: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"llm-{time.strftime('%Y%m%d')}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def record(node: str, kind: str, prompt: str, response: str, sec: float, model: str = None):
    """問答を1件、ログとメモリの両方へ残す。"""
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "req": getattr(_TL, "req", ""), "node": node,
           "kind": kind, "model": model or MODEL, "sec": round(sec, 2),
           "prompt": prompt, "response": response}
    _log(rec)
    exchanges().append({k: rec[k] for k in ("node", "kind", "model", "sec", "prompt", "response")})
    return rec


class LoggedChat:
    """ChatOllama を包み、invoke のたびに問答を記録する。

    呼び出し側を書き換えずに全件を拾うため、透過的な委譲にしている。
    どのノードからの呼び出しかは set_node() で切り替える。
    """

    def __init__(self, factory, kind: str = "text"):
        """factory(model名) -> クライアント。ノードごとに違うモデルを使うため遅延生成する。"""
        self._factory, self._kind, self._cache = factory, kind, {}

    def set_node(self, node: str):
        # 同じインスタンスを複数のリクエストが使うため、ノード名もスレッドごとに持つ
        _TL.node = node
        return self

    def _client(self, model: str):
        if model not in self._cache:
            self._cache[model] = self._factory(model)
        return self._cache[model]

    def invoke(self, prompt, *a, **kw):
        node = getattr(_TL, "node", "-")
        model = NODE_MODELS.get(node, MODEL)
        t0 = time.time()
        out = self._client(model).invoke(prompt, *a, **kw)
        text = getattr(out, "content", out)
        record(node, self._kind, prompt if isinstance(prompt, str) else str(prompt),
               str(text), time.time() - t0, model=model)
        return out


def model_sizes() -> dict:
    """Ollama が持っているモデルの容量を {モデル名: "4.7G"} で返す。

    画面に「どのノードがどれだけの大きさのモデルを使っているか」を出すため。
    取得できないときは空を返し、表示側は名前だけを出す。
    """
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=5) as r:
            data = json.loads(r.read())
    except Exception:
        return {}
    out = {}
    for m in data.get("models", []):
        # ollama list と同じ 10進GB で出す（CLI の表示と食い違わないため）
        # 単位は GB に固定する（M と G が混ざると比較しにくいため）
        gb = m.get("size", 0) / 1_000_000_000
        label = f"{gb:.1f}GB"
        out[m.get("name", "")] = label
        out[m.get("name", "").split(":")[0]] = label
    return out
