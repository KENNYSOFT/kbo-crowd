"""대시보드가 읽을 data.js 를 만든다.

python analysis/export_dashboard.py

dashboard/index.html 이 이 파일을 읽는다. 경기 원자료에 더해 검열 회귀가
추정한 티켓 파워와 잠재 수요를 함께 내보낸다. 화면에서 모델을 돌릴 수는
없으니 여기서 한 번 계산해 값으로 넘긴다.

산출물이라 버전 관리하지 않는다. 대시보드를 보려면 이 스크립트를 돌린 뒤
dashboard 디렉토리를 정적 서버로 열면 된다.
"""

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model
from demand import team_effects
from predict_today import grade, predict_games, resolve_target
from rain_price import rain_effect

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOW_ORDER = ["월", "화", "수", "목", "금", "토", "일"]


def rain_summary(df):
    """비가 수요에 주는 영향을 화면에 넘길 형태로 간추린다.

    3월은 뺀다. 기상청 시간자료는 11월부터 이듬해 3월까지 강수를 3시간 누적으로
    주므로 경기 시간대 강수가 어긋난다. rain_price.py 와 같은 기준이다.
    날씨가 아직 없으면 None 을 주어 화면이 그 섹션을 통째로 접게 한다.
    """
    if "rain_game" not in df.columns or df["rain_game"].notna().sum() == 0:
        return None

    d = df[df["month"].astype(str) != "03"].copy()
    d["rain_game"] = pd.to_numeric(d["rain_game"], errors="coerce")
    d["rain_day"] = pd.to_numeric(d.get("rain_day"), errors="coerce")
    d["is_dome"] = pd.to_numeric(d["is_dome"], errors="coerce").fillna(0)
    wet = d["rain_game"].fillna(0) > 0
    weekend = d["dow"].isin(["토", "일"])

    def pair(subset):
        w, dry = subset[subset["rain_game"].fillna(0) > 0], subset[subset["rain_game"].fillna(0) <= 0]
        if len(w) < 5 or len(dry) < 5:
            return None
        return {"games": len(subset), "wet": len(w),
                "wetOcc": round(float(w["occupancy"].mean()), 4),
                "dryOcc": round(float(dry["occupancy"].mean()), 4)}

    shelter = []
    for label, mask in [("야외 구장", d["is_dome"] == 0), ("돔 구장", d["is_dome"] == 1)]:
        got = pair(d[mask])
        if got:
            shelter.append(dict(label=label, **got))

    effects = []
    for label, subset in [("야외 전체", d[d["is_dome"] == 0]),
                          ("야외 주중", d[(d["is_dome"] == 0) & ~weekend]),
                          ("야외 주말", d[(d["is_dome"] == 0) & weekend])]:
        name, n, pct, _per_mm, n_wet, _why = rain_effect(subset, label)
        if pct is not None:
            effects.append({"label": name, "n": n, "wet": n_wet, "pct": round(pct, 4)})

    timing = []
    wet_day = d["rain_day"].fillna(0) > 0
    for label, mask in [("경기 중에 왔다", wet), ("그날만 왔다", wet_day & ~wet), ("안 왔다", ~wet_day)]:
        subset = d[mask]
        if len(subset):
            timing.append({"label": label, "n": len(subset),
                           "occ": round(float(subset["occupancy"].mean()), 4),
                           "sellout": round(float(subset["sold_out"].mean()), 4)})

    return {"shelter": shelter, "effects": effects, "timing": timing, "n": len(d)}


def next_day(min_season):
    """다음 경기일 예측. 화면 맨 위에 올릴 값이다.

    CLI 의 predict_today 와 같은 함수를 써서 두 화면의 숫자가 갈리지 않게 한다.
    예정 경기가 없거나(시즌 종료) 학습이 부족하면 None 이고, 화면은 그 섹션을
    통째로 접는다.
    """
    target, _moved = resolve_target()
    if target is None:
        return None
    games, meta = predict_games(target, min_season=min_season, weather=True)
    if games is None:
        return None

    # 강수확률은 모델 피처가 아니라 읽는 사람을 위한 값이라 예보에서 직접 읽는다.
    pop = {}
    path = os.path.join(ROOT, "data", "forecast.csv")
    if os.path.exists(path):
        fc = pd.read_csv(path, dtype={"hour": str})
        for r in fc.itertuples():
            pop[(r.stadium, r.date, str(r.hour).zfill(2))] = r.rain_prob

    def number(value, digits=1):
        return None if value is None or pd.isna(value) else round(float(value), digits)

    rows = []
    for g in games.itertuples():
        hour = str(g.start)[:2]
        note = getattr(g, "note", "")
        rows.append({
            "stadium": g.stadium,
            "start": g.start,
            "home": g.home,
            "away": g.away,
            "seats": int(g.capacity),
            "expected": int(round(g.expected)),
            "latent": int(round(g.latent)),
            "prob": round(float(g.prob), 4),
            "grade": grade(g.prob),
            "temp": number(getattr(g, "temp", None)),
            "rain": number(getattr(g, "rain_game", None)),
            "pop": number(pop.get((g.stadium, g.date, hour)), 0),
            "note": note if isinstance(note, str) and note not in ("-", "") else "",
        })

    return {
        "date": target,
        "weather": meta["weather"],
        "forecast": meta["forecast"],
        "trainN": meta["train"],
        "games": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(ROOT, "dashboard", "data.js"))
    ap.add_argument("--min-season", default="2023", help="모델 학습 시작 시즌")
    args = ap.parse_args()

    full = pd.read_csv(
        os.path.join(ROOT, "data", "dataset.csv"), dtype={"season": str, "month": str}
    )
    stadiums = pd.read_csv(os.path.join(ROOT, "data", "stadiums.csv"))
    home_of = dict(zip(stadiums["stadium"], stadiums["home_teams"].fillna("")))

    seasons = sorted(full["season"].unique())
    teams = sorted(set(full["home"]) | set(full["away"]))
    stds = sorted(full["stadium"].unique())

    games = [
        [
            seasons.index(r.season),
            r.date[5:],
            DOW_ORDER.index(r.dow),
            teams.index(r.home),
            teams.index(r.away),
            stds.index(r.stadium),
            int(r.crowd),
            int(r.capacity) if pd.notna(r.capacity) else 0,
        ]
        for r in full.itertuples()
    ]

    # 검열 회귀로 티켓 파워와 잠재 수요를 낸다.
    train = model.load_dataset(min_season=args.min_season)
    X = model.design_matrix(train)
    fit = model.fit_tobit(X.values, train["log_crowd"].values, train["log_cap"].values)
    all_teams = sorted(set(train["home"]) | set(train["away"]))
    train = train.assign(latent=np.exp(model.latent_demand(fit, X.values)))

    by_home = train.groupby("home").agg(
        observed=("crowd", "mean"), latent=("latent", "mean"),
        sellout=("sold_out", "mean"), seats=("capacity", "median"),
    )

    payload = {
        "next": next_day(args.min_season),
        "rain": rain_summary(train),
        "generated": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST"),
        "lastGame": full["date"].max(),
        "seasons": seasons,
        "teams": teams,
        "stadiums": stds,
        "dows": DOW_ORDER,
        "stadiumHome": {s: home_of.get(s, "") for s in stds},
        "g": games,
        "model": {
            "minSeason": args.min_season,
            "n": fit["n"],
            "censored": fit["n_censored"],
            "sigma": round(fit["sigma"], 4),
            "homePower": {k: round(v, 3) for k, v in team_effects(fit, X, "hometeam", all_teams).items()},
            "awayPower": {k: round(v, 3) for k, v in team_effects(fit, X, "awayteam", all_teams).items()},
            "demand": {
                team: {
                    "observed": int(r.observed),
                    "latent": int(r.latent),
                    "sellout": round(float(r.sellout), 4),
                    "seats": int(r.seats),
                }
                for team, r in by_home.iterrows()
            },
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    text = "window.KBO=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";"
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print("%s  %d경기 %.0fKB" % (args.out, len(games), len(text.encode("utf-8")) / 1024))


if __name__ == "__main__":
    main()
