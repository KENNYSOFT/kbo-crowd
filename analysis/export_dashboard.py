"""대시보드가 읽을 data.js 를 만든다.

python analysis/export_dashboard.py

dashboard/index.html 이 이 파일을 읽는다. 경기 원자료에 더해 검열 회귀가
추정한 티켓 파워와 잠재 수요를 함께 내보낸다. 화면에서 모델을 돌릴 수는
없으니 여기서 한 번 계산해 값으로 넘긴다.

산출물이라 버전 관리하지 않는다. 대시보드를 보려면 이 스크립트를 돌린 뒤
dashboard 디렉토리를 정적 서버로 열면 된다.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model
from demand import team_effects

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOW_ORDER = ["월", "화", "수", "목", "금", "토", "일"]


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
