# LangChainViewer

**ローカルLLMだけで動く「資料検索＋SQL集計」エージェントと、その動作を1手ずつ見せるビューア。**

エージェントは中で何をしているか見えません。このリポジトリは、**見えるようにしたら
何が分かったか**を残したものです。同じ課題を LangGraph / CrewAI / AutoGen の3基盤で実装し、
共有層を固定して同一条件で比較しています。

推論も埋め込みもローカル（Ollama）で完結し、**外部APIの費用は0円**です。

- 公開ページ: https://tobisako.github.io/LangChainViewer/
- ライセンス: MIT

## 何ができるか

| | |
|---|---|
| **実況ビュー** | グラフのノードが実行順に点灯し、LLMとの問答が毎回そのまま出る。標準ライブラリのみ（追加依存ゼロ） |
| **資料検索** | ベクトル類似度0.7＋語一致0.3のハイブリッド検索。ベクトルDBは立てない（`.npz` 1本） |
| **SQL集計** | 読み取り専用で開き、`SELECT` / `WITH` 以外は実行しない |
| **安全機構** | 入力の遮断 → 行動分類による許可制 → 出力検査 の3層 |
| **ノード別モデル** | 5つの処理に別々のモデルを割り当てられる。0.8GB〜54GBの構成を切り替えて比較できる |
| **比較ベンチ** | 3基盤を同一の未見セットで測る。採点器も共有 |

## 使うもの

- Python 3.12 以上
- [Ollama](https://ollama.com/)（ローカル推論）

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

## 準備

```bash
git clone https://github.com/tobisako/LangChainViewer.git
cd LangChainViewer

python3 -m venv .venv-langgraph
.venv-langgraph/bin/pip install -r requirements-langgraph.txt

python3 tools/make_sample_data.py          # サンプルデータを生成する
.venv-langgraph/bin/python tools/build_index.py   # 資料の索引を作る
```

### サンプルデータについて

`data/` の中身は **`tools/make_sample_data.py` がその場で乱数生成したもの**です。
外部から持ち込んだデータは1件も含まれません。乱数の種を固定しているので、
**誰が実行しても同じデータ・同じ集計値**になり、下の測定結果を再現できます。

個人を特定しうる列（`zip_code` / `birthday` / `latitude` / `longitude` / `ma_lead_id`）は
**意図的に残してあります**。安全機構が働くかどうかを試すために必要だからです。

## 動かす

```bash
.venv-langgraph/bin/python viewer/server.py
```

| URL | 何が見えるか |
|---|---|
| http://127.0.0.1:8765/ | 質問を入れて結果を見る |
| http://127.0.0.1:8765/live | **実況。ノードが順に点灯し、LLMとの問答が流れる** |
| http://127.0.0.1:8765/live?i=3 | シナリオを1本だけ繰り返す |

## 比較ベンチを回す

3基盤は依存が同居できないので、venv を分けます。

```bash
.venv-langgraph/bin/python run_bench.py langgraph holdout
.venv-langgraph/bin/python run_bench.py langgraph safety
```

結果は `results/*.csv` に出ます。全問の判定・SQL・応答時間つきです。

## モデル構成を変える

```bash
POC_PROFILE=large .venv-langgraph/bin/python run_bench.py langgraph holdout
```

| 構成 | plan | retrieve | query | critique | report |
|---|---|---|---|---|---|
| `small`（既定） | qwen2.5:1.5b | llama3.2:1b | qwen2.5-coder:1.5b | gemma3:1b | qwen2.5:7b |
| `large` | 〃 | 〃 | qwen3:32b | qwen3:32b | qwen2.5:72b-instruct-q5_K_M |
| `mixed` | 〃 | 〃 | qwen3:32b | dolphin-mixtral:8x7b | qwen2.5:72b-instruct-q5_K_M |
| `uncensored` | 〃 | 〃 | qwen2.5-coder:1.5b | gemma3:1b | dolphin-mixtral:8x7b |

`uncensored` を用意しているのは、**安全機構がモデル側の拒否に依存していないことを確かめるため**です。
拒否調整を外したモデルを回答生成に置いても安全セットが通るなら、
守っているのはモデルではなくエージェント側の3層だと言えます。

`shared/models.py` で手元のモデル一覧、`shared/smoke.py` で全モデルの疎通を確認できます。

## 構成

```
viewer/       実況ビューア（標準ライブラリのみ）
agents/       langgraph / crewai / autogen の3実装
shared/       共有層。core（データ・検索・SQL・安全）・judge（採点器）・evals（評価セット）
tools/        サンプルデータ生成・索引作成
docs/         公開ページ
```

`shared/` に置いたものは**3基盤で同一**です。フレームワークが決めてよいのは
オーケストレーションの切り方・プロンプト・再検索の条件・レポートの書式だけで、
データ・索引・モデル・安全実装・評価セット・採点器は共有層に固定しています。
これを守らないと、比較しているのがフレームワークなのかチューニングなのか分からなくなります。
