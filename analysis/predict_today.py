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


WEATHER_COLS = ["temp", "rain_game", "rain_day", "humid", "wind", "cloud"]


def data(name):
    return pd.read_csv(os.path.join(ROOT, "data", name), dtype={"season": str, "month": str})


def attach_forecast(games, span=3):
    """data/forecast.csv 의 예보를 경기 행에 붙인다.

    관측 쪽과 컬럼 이름을 맞춰야 모델이 같은 자리로 받는다. 경기 시간대 강수는
    관측과 같은 방식으로 시작 시각부터 span 시간을 더하고, 그날 누적은 그 날짜
    예보 전체를 더한다. 예보가 없는 구장이나 시각은 비워 두어 호출한 쪽이
    평년값으로 채우게 한다.
    """
    path = os.path.join(ROOT, "data", "forecast.csv")
    if not os.path.exists(path):
        return games, 0
    fc = pd.read_csv(path, dtype={"hour": str})
    if fc.empty:
        return games, 0

    for col in ["temp", "rain_mm", "humid", "wind", "sky"]:
        fc[col] = pd.to_numeric(fc[col], errors="coerce")
    at = {(r.stadium, r.date, int(r.hour)): r for r in fc.itertuples()}
    day_rain = fc.groupby(["stadium", "date"])["rain_mm"].sum()

    values = {col: [] for col in WEATHER_COLS}
    matched = 0
    for g in games.itertuples():
        start = str(getattr(g, "start", "") or "")
        hour = int(start[:2]) if len(start) >= 2 and start[:2].isdigit() else None
        row = at.get((g.stadium, g.date, hour)) if hour is not None else None
        if row is None:
            for col in WEATHER_COLS:
                values[col].append(float("nan"))
            continue
        matched += 1
        total, seen = 0.0, False
        for offset in range(span):
            nxt = at.get((g.stadium, g.date, (hour + offset) % 24))
            if nxt is not None and pd.notna(nxt.rain_mm):
                total += float(nxt.rain_mm)
                seen = True
        values["temp"].append(row.temp)
        values["rain_game"].append(total if seen else float("nan"))
        values["rain_day"].append(float(day_rain.get((g.stadium, g.date), float("nan"))))
        values["humid"].append(row.humid)
        values["wind"].append(row.wind)
        values["cloud"].append(row.sky)

    games = games.copy()
    for col in WEATHER_COLS:
        games[col] = values[col]
    return games, matched


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


def resolve_target(date=None):
    """예보할 날짜를 정한다. 그 날짜에 경기가 없으면 다음 경기일로 넘긴다.

    (날짜, 넘어갔는지) 를 돌려주고, 남은 경기가 없으면 (None, True).
    """
    target = date or dt.date.today().isoformat()
    if not upcoming(target).empty:
        return target, False
    schedule = data("schedule.csv")
    later = sorted(d for d in schedule["date"].unique() if d > target)
    return (later[0], True) if later else (None, True)


def predict_games(target, min_season="2023", weather=True, log=None):
    """그 날짜 경기의 매진 확률과 예상 관중을 낸다.

    games 에 capacity, prob, expected, latent 를 붙여 돌려준다. 낼 수 없으면
    (None, 사유) 다. CLI 와 대시보드가 같은 함수를 쓰게 해서, 한쪽만 고쳐
    두 화면의 숫자가 갈리는 일이 없게 한다.
    """
    def say(message):
        if log:
            log(message)

    games = upcoming(target)
    if games.empty:
        return None, {"reason": "%s 에 경기가 없다" % target}

    train = model.load_dataset(min_season=min_season)
    train = train[train["date"] < target]
    if len(train) < 200:
        return None, {"reason": "학습할 경기가 %d개뿐이다" % len(train)}

    if weather and train["temp"].notna().sum() == 0:
        say("날씨 데이터가 비어 있어 날씨 없이 예보한다.")
        weather = False

    # 그 시즌 그 구장의 상한. 아직 안 열린 경기라 관측값에서 직접 못 구한다.
    caps = train.groupby(["season", "stadium"])["capacity"].max()
    season = games["season"].astype(str).iloc[0]
    games["capacity"] = [
        caps.get((season, s), caps.get((str(int(season) - 1), s), np.nan))
        for s in games["stadium"]
    ]
    games = games[games["capacity"].notna()].copy()
    if games.empty:
        return None, {"reason": "상한을 알 수 없는 구장뿐이다"}

    matched = 0
    if weather:
        # 아직 열리지 않은 경기에는 관측이 없다. 그대로 두면 design_matrix 가
        # 그 열을 만들지 못하고 reindex 가 0 으로 채우는데, 그것은 '결측'이
        # 아니라 기온 0 도에 습도 0 퍼센트라는 뜻이 되어 예측이 조용히 틀어진다.
        # 그래서 예보를 먼저 붙이고, 그래도 빈 자리만 평년값으로 둔다.
        games, matched = attach_forecast(games)
        if matched:
            say("  기상청 단기예보를 붙였다 (%d경기)." % matched)
        missing = [c for c in WEATHER_COLS
                   if c not in games.columns or games[c].isna().all()]
        for col in missing:
            games[col] = pd.to_numeric(train[col], errors="coerce").median()
        partial = [c for c in WEATHER_COLS
                   if c in games.columns and games[c].isna().any() and c not in missing]
        for col in partial:
            games[col] = games[col].fillna(pd.to_numeric(train[col], errors="coerce").median())
        if missing:
            say("  예보가 없어 %s 는 평년값으로 둔다." % ", ".join(missing))
        elif partial:
            say("  일부 경기의 %s 가 비어 평년값으로 채웠다." % ", ".join(partial))

    X = model.design_matrix(train, weather=weather)
    fit = model.fit_tobit(X.values, train["log_crowd"].values, train["log_cap"].values)
    ref = [c for c in X.columns if c != "const"]

    Xg = model.design_matrix(games, weather=weather, reference=ref)
    log_cap = np.log(games["capacity"].values)
    games["prob"] = model.sellout_probability(fit, Xg.values, log_cap)
    games["expected"] = np.exp(model.expected_observed(fit, Xg.values, log_cap))
    games["latent"] = np.exp(model.latent_demand(fit, Xg.values))
    games = games.sort_values("prob", ascending=False).reset_index(drop=True)

    return games, {"target": target, "train": len(train),
                   "weather": bool(weather), "forecast": int(matched)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="예보할 날짜. 비우면 오늘, 오늘 경기가 없으면 다음 경기일")
    ap.add_argument("--weather", action="store_true", help="날씨 변수를 함께 쓴다")
    ap.add_argument("--min-season", default="2023")
    args = ap.parse_args()

    target, moved = resolve_target(args.date)
    if target is None:
        raise SystemExit("%s 이후 예정 경기가 없다." % (args.date or "오늘"))
    if moved:
        print("%s 에는 경기가 없어 다음 경기일 %s 로 본다." % (args.date or "오늘", target))

    games, meta = predict_games(target, args.min_season, args.weather, log=print)
    if games is None:
        raise SystemExit(meta["reason"])

    print()
    print("%s 티켓 난이도  (학습 %d경기, %s 이전)" % (target, meta["train"], target))
    print("-" * 78)
    print("  %-5s %-4s %-11s %6s %7s %7s  %5s  %s"
          % ("구장", "시각", "경기", "좌석", "예상", "잠재수요", "매진율", "난이도"))
    for g in games.itertuples():
        note = getattr(g, "note", "")
        cancel = " [%s]" % note if isinstance(note, str) and note not in ("-", "") else ""
        print("  %-5s %-5s %-4s vs %-4s %6d %7d %7d  %4.0f%%  %s%s"
              % (g.stadium, g.start, g.away, g.home,
                 g.capacity, g.expected, g.latent, g.prob * 100,
                 grade(g.prob), cancel))
    print("-" * 78)
    print("  예상은 좌석 제약까지 반영한 관중 수, 잠재수요는 제약이 없다면의 수요다.")
    print("  둘이 벌어질수록 표 구하기가 어렵다는 뜻이다.")


if __name__ == "__main__":
    main()
