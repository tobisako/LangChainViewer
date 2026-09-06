#!/usr/bin/env python3
"""このリポジトリで使うサンプルデータを、その場で機械生成する。

**外部から持ち込んだデータは1件も含まれない。** すべてこのスクリプトが乱数で作る。
seed を固定してあるので、誰が実行しても同じデータ・同じ集計値になる。

作るもの:
  data/sample.sqlite   3テーブル（lead_plot_out / lead_plot_agg / m_property）
  data/*.csv           上の3テーブルのCSV
  data/corpus/*.md     エージェントが検索する資料2本

使い方:
  python tools/make_sample_data.py
"""
import csv, os, random, sqlite3, datetime as dt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SEED = 20260906
N_LEAD, N_PROP = 6689, 300

PREFS = [
    ("北海道", 3), ("宮城県", 2), ("東京都", 12), ("神奈川県", 8), ("埼玉県", 9),
    ("千葉県", 6), ("愛知県", 7), ("静岡県", 3), ("大阪府", 8), ("京都府", 3),
    ("兵庫県", 6), ("広島県", 3), ("福岡県", 5), ("熊本県", 2), ("沖縄県", 2),
]
MEDIA = [
    ("折込チラシ", 15), ("Web検索", 22), ("紹介", 12), ("SNS広告", 14),
    ("現地看板", 9), ("住宅情報誌", 8), ("テレビCM", 5), ("その他", 4),
]
CITIES = ["中央区", "北区", "南区", "緑区", "港区", "西区", "東区", "青葉区", "泉区", "若葉区"]
TOWNS = ["本町", "栄町", "緑ケ丘", "桜台", "旭町", "宮前", "花見川", "松原", "上野台", "新川"]


def pick(weighted, rnd):
    total = sum(w for _, w in weighted)
    x = rnd.uniform(0, total)
    for v, w in weighted:
        x -= w
        if x <= 0:
            return v
    return weighted[-1][0]


def d(rnd, y0, y1):
    base = dt.date(y0, 1, 1)
    return (base + dt.timedelta(days=rnd.randrange((dt.date(y1, 12, 31) - base).days))).isoformat()


def make_properties(rnd):
    rows = []
    for i in range(1, N_PROP + 1):
        pref = pick(PREFS, rnd)
        lo = rnd.randrange(1800, 3600, 50)
        rows.append({
            "object_code": f"P{i:04d}",
            "formal_object_name": f"{pref}{rnd.choice(TOWNS)}{rnd.randrange(1, 40)}期",
            "prefecture": pref,
            "city": rnd.choice(CITIES),
            "budget_min_man": float(lo),
            "budget_max_man": float(lo + rnd.randrange(300, 1500, 50)),
            # 建物面積。最小値が評価問題の答えになるので、下限は明示的に置く
            "building_area_avg": float(rnd.randrange(45, 165)),
            "latitude": round(rnd.uniform(26.2, 45.4), 6),
            "longitude": round(rnd.uniform(127.7, 145.6), 6),
        })
    rows[0]["building_area_avg"] = 45.0   # 最小値を確定させる
    return rows


def make_leads(rnd, props):
    rows = []
    for i in range(1, N_LEAD + 1):
        p = rnd.choice(props)
        req = d(rnd, 2023, 2026)
        reaction = 1.0 if rnd.random() < 0.62 else 0.0
        visitor = 1.0 if reaction and rnd.random() < 0.55 else 0.0
        contractor = 1.0 if visitor and rnd.random() < 0.31 else 0.0
        rows.append({
            "pid": f"L{i:06d}",
            "request_date": req,
            "first_visit_date": d(rnd, 2023, 2026) if visitor else "",
            "address1": p["prefecture"],
            "address2": rnd.choice(CITIES),
            "address3": rnd.choice(TOWNS),
            "address4_1": f"{rnd.randrange(1, 40)}-{rnd.randrange(1, 30)}",
            "zip_code": f"{rnd.randrange(100, 999)}-{rnd.randrange(1000, 9999)}",
            "birthday": d(rnd, 1955, 2002),
            "recognition_media": pick(MEDIA, rnd),
            "recognition_media_other": "",
            "budget_price": float(rnd.randrange(1500, 6500, 50)),
            "object_code": p["object_code"],
            "formal_object_name": p["formal_object_name"],
            "hope_building_area": float(rnd.randrange(40, 130)),
            "register_date": req,
            "subscription_date": d(rnd, 2023, 2026) if contractor else "",
            "contract_date": d(rnd, 2023, 2026) if contractor else "",
            "cancel_contract_date": "",
            "reaction_flg": reaction,
            "visitor_flg": visitor,
            "contractor_flg": contractor,
            "contract_generations": float(rnd.choice([20, 30, 40, 50, 60, 70])),
            "visit_flg_current_month": 1.0 if visitor and rnd.random() < 0.08 else 0.0,
            "contract_flg_current_month": 1.0 if contractor and rnd.random() < 0.09 else 0.0,
            "latitude": round(rnd.uniform(26.2, 45.4), 6),
            "longitude": round(rnd.uniform(127.7, 145.6), 6),
            "budget_band": rnd.choice(["1500-2500", "2500-3500", "3500-4500", "4500-"]),
            "duplicated_flag_ma": 1.0 if rnd.random() < 0.04 else 0.0,
            "ma_lead_id": f"MA{rnd.randrange(10**7, 10**8)}",
        })
    return rows


def write(conn, table, rows):
    cols = list(rows[0].keys())
    # 数値列は REAL で作る。TEXT のままだと AVG / MIN が文字列比較になり集計できない。
    defs = ", ".join(f'"{c}" {"REAL" if isinstance(rows[0][c], float) else "TEXT"}' for c in cols)
    conn.execute(f'DROP TABLE IF EXISTS "{table}"')
    conn.execute(f'CREATE TABLE "{table}" ({defs})')
    conn.executemany(f'INSERT INTO "{table}" VALUES ({",".join("?"*len(cols))})',
                     [[r[c] for c in cols] for r in rows])
    with open(os.path.join(DATA, f"{table}.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def main():
    os.makedirs(DATA, exist_ok=True)
    rnd = random.Random(SEED)
    props = make_properties(rnd)
    leads = make_leads(rnd, props)
    db = os.path.join(DATA, "sample.sqlite")
    if os.path.exists(db):
        os.remove(db)
    conn = sqlite3.connect(db)
    write(conn, "m_property", props)
    write(conn, "lead_plot_out", leads)
    write(conn, "lead_plot_agg", leads)   # 集計用。出力用と同一構造・同一件数
    conn.commit(); conn.close()
    print(f"  data/sample.sqlite  lead_plot_out={len(leads)} / lead_plot_agg={len(leads)} / m_property={len(props)}")


if __name__ == "__main__":
    main()
