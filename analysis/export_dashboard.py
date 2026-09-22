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
from predict_today import grade, predict_games, upcoming_dates
from rain_price import BANDS, BAND_LABELS, rain_effect, scheduled_with_rain

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


def rain_bands(min_season, keep_march=False):
    """강수 구간마다 경기가 열릴 확률과 그때의 점유율을 함께 낸다.

    관중 기록만 보면 비의 큰 몫이 안 보인다. 비가 셀수록 경기가 아예 취소되어
    그 경기가 표에서 통째로 빠지기 때문이다. 그래서 일정에서 출발해 취소를
    관중 0 으로 놓고 편성 한 경기당 기대 점유율을 낸다.
    """
    sch = scheduled_with_rain(min_season, keep_march)
    if sch.empty:
        return None
    sch = sch.copy()
    sch["band"] = pd.cut(sch["rain"], bins=BANDS, labels=BAND_LABELS)

    rows = []
    for band, g in sch.groupby("band", observed=True):
        opened = g[~g["cancelled"] & g["occupancy"].notna()]
        if not len(opened):
            continue
        rate = float(g["cancelled"].mean())
        occ = float(opened["occupancy"].mean())
        rows.append({
            "label": str(band),
            "games": int(len(g)),
            "cancelled": int(g["cancelled"].sum()),
            "cancelRate": round(rate, 4),
            "opened": int(len(opened)),
            "openOcc": round(occ, 4),
            "expectedOcc": round((1 - rate) * occ, 4),
        })
    if len(rows) < 2:
        return None

    wet = sch[sch["rain"] > 0]
    return {"rows": rows, "wetGames": int(len(wet)),
            "wetCancelled": int(wet["cancelled"].sum())}


def forecast_display():
    """화면에 곁들일 예보를 (구장, 날짜, 시각) 으로 찾을 수 있게 편다.

    모델 입력이 아니라 읽는 사람을 위한 값이다. 날씨는 매진 예측을 나아지게
    하지 않아서 모델에서 뺐지만(analysis/lead_time.py), 당일 관중은 분명히
    깎으므로 표를 이미 쥔 사람에게는 여전히 볼 값이다. 단기예보라 앞 사흘만
    차고 나머지는 빈다.
    """
    path = os.path.join(ROOT, "data", "forecast.csv")
    if not os.path.exists(path):
        return {}
    fc = pd.read_csv(path, dtype={"hour": str})
    return {(r.stadium, r.date, str(r.hour).zfill(2)): r for r in fc.itertuples()}


def forecast_sections(min_season, lead=7, span=2):
    """화면 맨 위 두 절에 쓸 값을 낸다. 두 절이 묻는 것은 서로 다르다.

    오늘 경기는 표가 이미 팔렸거나 당일권만 남아서 '구할 수 있나' 를 물을
    자리가 아니고, 그날 몇 명이 오느냐가 관심사다. 반대로 지금 막 예매가
    열리는 날짜는 아직 살 수 있으니 매진 확률이 답할 질문이다.

    그 사이 날짜(내일부터 lead 일 전까지)는 넣지 않는다. 이미 표가 풀려 있어
    지금 결정할 것이 없기 때문이다. 그래서 창은 오늘 하루와 lead 일 뒤부터
    span 일이고, 기본값 7과 2는 티켓이 대체로 경기 일주일 전에 열리는 것을
    따른 것이다.

    CLI 의 predict_today 와 같은 함수를 쓰고 한 번의 학습으로 두 갈래를 모두
    내므로 두 절의 숫자가 서로 어긋날 수 없다. 예정 경기가 없거나(시즌 종료)
    학습이 부족하면 None 이고, 화면은 그 절을 통째로 접는다.
    """
    today = dt.date.today().isoformat()
    dates = sorted(set(upcoming_dates(1, today) + upcoming_dates(span, today, lead=lead)))
    if not dates:
        return None
    games, meta = predict_games(dates, min_season=min_season, weather=False)
    if games is None:
        return None

    at = forecast_display()

    def number(value, digits=1):
        return None if value is None or pd.isna(value) else round(float(value), digits)

    grouped = []
    for date, day in games.groupby("date", sort=True):
        rows = []
        for g in day.itertuples():
            fc = at.get((g.stadium, g.date, str(g.start)[:2]))
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
                "temp": number(getattr(fc, "temp", None)) if fc is not None else None,
                "pop": number(getattr(fc, "rain_prob", None), 0) if fc is not None else None,
                "note": note if isinstance(note, str) and note not in ("-", "") else "",
            })
        grouped.append({"date": date, "dow": day["dow"].iloc[0], "games": rows})

    today_block = grouped.pop(0) if grouped and grouped[0]["date"] == today else None

    return {
        "today": today_block,
        "from": grouped[0]["date"] if grouped else None,
        "to": grouped[-1]["date"] if grouped else None,
        "trainN": meta["train"],
        "days": grouped,
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
        "forecast": forecast_sections(args.min_season),
        "rain": rain_summary(train),
        "rainBands": rain_bands(args.min_season),
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
