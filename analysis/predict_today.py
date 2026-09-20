"""오늘 경기의 티켓 난이도를 예보한다.

python analysis/predict_today.py [--date YYYY-MM-DD] [--weather]

그 날짜 이전 경기만으로 모델을 맞춘 뒤 그날 경기의 매진 확률과 예상 관중을
낸다. 학습에 그날 이후가 섞이면 실제로는 알 수 없는 것을 아는 셈이 되므로
날짜로 자른다.

예정 경기의 순위와 연승은 build_standings.py 가 미리 붙여 둔 값을 쓴다.
"""

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRADES = [
    (0.80, "매우 어려움"),
    (0.60, "어려움"),
    (0.40, "보통"),
    (0.20, "여유"),
    (0.00, "넉넉"),
]


def grade(p):
    for threshold, label in GRADES:
        if p >= threshold:
            return label
    return GRADES[-1][1]


def data(name):
    return pd.read_csv(os.path.join(ROOT, "data", name), dtype={"season": str, "month": str})


def upcoming(date):
    """그 날짜의 경기를 순위, 구장 메타와 함께 조립한다."""
    schedule = data("schedule.csv")
    standings = data("standings.csv")
    stadiums = data("stadiums.csv")

    games = schedule[schedule["date"] == date].copy()
    if games.empty:
        return games

    games = games.merge(
        standings.drop(columns=["stadium"]), on=["date", "home", "away"], how="left"
    )
    games = games.merge(stadiums, on="stadium", how="left")
    games["dow"] = pd.to_datetime(games["date"]).dt.strftime("%a").map(
        {"Mon": "월", "Tue": "화", "Wed": "수", "Thu": "목",
         "Fri": "금", "Sat": "토", "Sun": "일"}
    )
    games["month"] = games["date"].str[5:7]
    return games


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="예보할 날짜. 비우면 오늘, 오늘 경기가 없으면 다음 경기일")
    ap.add_argument("--weather", action="store_true", help="날씨 변수를 함께 쓴다")
    ap.add_argument("--min-season", default="2023")
    args = ap.parse_args()

    target = args.date or dt.date.today().isoformat()
    games = upcoming(target)
    if games.empty:
        schedule = data("schedule.csv")
        later = sorted(d for d in schedule["date"].unique() if d > target)
        if not later:
            raise SystemExit("%s 이후 예정 경기가 없다." % target)
        target = later[0]
        games = upcoming(target)
        print("%s 에는 경기가 없어 다음 경기일 %s 로 본다." % (args.date or "오늘", target))

    train = model.load_dataset(min_season=args.min_season)
    train = train[train["date"] < target]
    if len(train) < 200:
        raise SystemExit("학습할 경기가 %d개뿐이다." % len(train))

    if args.weather and train["temp"].notna().sum() == 0:
        print("날씨 데이터가 비어 있어 날씨 없이 예보한다.")
        args.weather = False

    X = model.design_matrix(train, weather=args.weather)
    fit = model.fit_tobit(X.values, train["log_crowd"].values, train["log_cap"].values)
    ref = [c for c in X.columns if c != "const"]

    # 그 시즌 그 구장의 상한. 아직 안 열린 경기라 관측값에서 직접 못 구한다.
    caps = train.groupby(["season", "stadium"])["capacity"].max()
    season = games["season"].astype(str).iloc[0]
    games["capacity"] = [
        caps.get((season, s), caps.get((str(int(season) - 1), s), np.nan))
        for s in games["stadium"]
    ]
    games = games[games["capacity"].notna()].copy()
    if games.empty:
        raise SystemExit("상한을 알 수 없는 구장뿐이라 예보할 수 없다.")

    Xg = model.design_matrix(games, weather=args.weather, reference=ref)
    log_cap = np.log(games["capacity"].values)
    prob = model.sellout_probability(fit, Xg.values, log_cap)
    expected = np.exp(model.expected_observed(fit, Xg.values, log_cap))
    latent = np.exp(model.latent_demand(fit, Xg.values))

    print()
    print("%s 티켓 난이도  (학습 %d경기, %s 이전)" % (target, len(train), target))
    print("-" * 78)
    print("  %-5s %-4s %-11s %6s %7s %7s  %5s  %s"
          % ("구장", "시각", "경기", "좌석", "예상", "잠재수요", "매진율", "난이도"))
    order = np.argsort(-prob)
    for i in order:
        g = games.iloc[i]
        note = g.get("note", "")
        cancel = " [%s]" % note if isinstance(note, str) and note not in ("-", "") else ""
        print("  %-5s %-5s %-4s vs %-4s %6d %7d %7d  %4.0f%%  %s%s"
              % (g["stadium"], g["start"], g["away"], g["home"],
                 g["capacity"], expected[i], latent[i], prob[i] * 100,
                 grade(prob[i]), cancel))
    print("-" * 78)
    print("  예상은 좌석 제약까지 반영한 관중 수, 잠재수요는 제약이 없다면의 수요다.")
    print("  둘이 벌어질수록 표 구하기가 어렵다는 뜻이다.")


if __name__ == "__main__":
    main()
