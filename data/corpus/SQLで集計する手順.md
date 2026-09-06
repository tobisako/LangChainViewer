# SQLで集計する手順

`data/sample.sqlite` に対して読み取り専用のSQLを実行し、集計結果を得る手順です。

## 0. この手順で確かめること

| | 操作 | 結論 |
|---|---|---|
| 1 | SQLite でSQLを実行する | 6,689件が数行の集計表になる |
| 2 | `sample.sqlite` を別フォルダへコピーする | 1ファイルなのでコピーだけで移せる |
| 3 | コピー先で同じSQLを実行する | 元と同じ結果が出る |

## 1. 接続

```bash
sqlite3 data/sample.sqlite
```

このリポジトリのエージェントは、同じファイルを **読み取り専用（`mode=ro`）** で開きます。
`SELECT` と `WITH` 以外は実行しません。

## 2. 件数を数える

```sql
SELECT COUNT(*) FROM lead_plot_agg WHERE address1 = '埼玉県';
```

## 3. 区分ごとに集計する

```sql
SELECT recognition_media, COUNT(*) AS c
FROM lead_plot_agg
GROUP BY recognition_media
ORDER BY c DESC;
```

## 4. 物件マスタと結合する

```sql
SELECT p.prefecture, COUNT(*) AS c
FROM lead_plot_agg AS l
JOIN m_property AS p ON l.object_code = p.object_code
GROUP BY p.prefecture;
```

結合に使う列は **`object_code`** です。

## 5. やらないこと

- `SELECT *` は使いません。個人を特定しうる列がまとめて出てしまうためです。
- `INSERT` / `UPDATE` / `DELETE` / `DROP` は実行しません。読み取り専用で開いています。
