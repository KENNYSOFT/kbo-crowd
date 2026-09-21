"""수집한 CSV 를 하나로 합쳐 모델 입력 data/dataset.csv 를 만든다.

관중, 일정, 순위, 날씨, 구장 메타를 (날짜, 홈, 원정) 으로 잇는다. 구장만으로는
더블헤더에서 두 경기가 한 행으로 겹치므로 팀까지 키에 넣는다.

여기서 수용인원과 매진 여부도 정한다. KBO 는 구장 수용인원을 이 데이터에
같이 주지 않기 때문에 관측값에서 되찾아야 한다. 판단 근거는 같은 값의 반복이다.
표가 다 팔리면 발표 관중 수가 정확히 같은 숫자로 여러 번 찍히고, 그렇게 두 번
이상 나온 값 중 최댓값이 그 시즌 그 구장의 상한이다. 시즌마다 따로 잡는 이유는
상한이 해마다 움직이기 때문이다. 대전은 2025년에 신구장으로 옮겼고, 사직과
창원은 좌석이 해마다 바뀌었으며, 2021~2022년은 코로나 입장 제한이 상한이었다.
"""

import argparse
import datetime as dt
from collections import Counter, defaultdict

import kbo

SELLOUT_RATIO = 0.98  # 상한의 98% 이상이면 매진으로 본다. 발표 수치가 조금씩 흔들린다.

HEADER = [
    "season", "date", "dow", "month", "start",
    "home", "away", "stadium", "home_teams", "is_dome", "is_secondary",
    "crowd", "capacity", "occupancy", "sold_out",
    "home_rank", "home_wpct", "home_gb", "home_streak", "home_last10",
    "away_rank", "away_wpct", "away_gb", "away_streak", "away_last10",
    "gb_from_playoff_line", "season_progress",
    "temp", "rain", "rain_game", "rain_day", "humid", "wind", "cloud",
]


def estimate_capacity(games):
    """(시즌, 구장) 별 수용인원을 관측값에서 추정한다."""
    buckets = defaultdict(list)
    for r in games:
        buckets[(r["season"], r["stadium"])].append(int(r["crowd"]))
    caps = {}
    for key, values in buckets.items():
        repeated = [v for v, n in Counter(values).items() if n >= 2]
        caps[key] = max(repeated) if repeated else max(values)
    return caps


def weather_index(rows):
    """(지점, 'YYYY-MM-DD HH') 로 찾을 수 있게 시간별 관측을 편다."""
    index = {}
    for r in rows:
        tm = r["tm"]
        if len(tm) >= 13:
            index[(r["stn"], tm[:13])] = r
    return index


def game_rain(index, stn, date, hour, span=3):
    """경기 시간대 강수량 합계. 시작 시각부터 span 시간.

    관측이 하나도 없으면 빈 값을 준다. 0 으로 채우면 '비가 오지 않았다'와
    '재지 못했다'가 같아져 버린다. 관측이 있는 시각만 더한다.
    """
    total = 0.0
    seen = False
    for offset in range(span):
        key = (stn, "%s %02d" % (date, (hour + offset) % 24))
        row = index.get(key)
        if row and row.get("rain") not in (None, ""):
            try:
                total += float(row["rain"])
                seen = True
            except ValueError:
                pass
    return round(total, 1) if seen else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    games = kbo.read_csv(kbo.data_path("games.csv"))
    if not games:
        raise SystemExit("data/games.csv 가 없다. 먼저 fetch_crowd.py 를 실행할 것.")
    schedule = kbo.read_csv(kbo.data_path("schedule.csv"))
    standings = kbo.read_csv(kbo.data_path("standings.csv"))
    stadiums = kbo.read_csv(kbo.data_path("stadiums.csv"))
    weather = weather_index(kbo.read_csv(kbo.data_path("weather.csv")))

    sched_by = {(r["date"], r["home"], r["away"]): r for r in schedule}
    stand_by = {(r["date"], r["home"], r["away"]): r for r in standings}
    meta_by = {r["stadium"]: r for r in stadiums}
    caps = estimate_capacity(games)

    unmatched_schedule = 0
    rows = []
    for g in games:
        key = (g["date"], g["home"], g["away"])
        sched = sched_by.get(key, {})
        stand = stand_by.get(key, {})
        meta = meta_by.get(g["stadium"], {})
        if not sched:
            unmatched_schedule += 1

        crowd = int(g["crowd"])
        cap = caps.get((g["season"], g["stadium"]), 0)
        occupancy = round(crowd / cap, 4) if cap else ""
        sold_out = 1 if cap and crowd >= cap * SELLOUT_RATIO else 0

        start = sched.get("start", "")
        stn = meta.get("asos_stn", "")
        hour = int(start[:2]) if len(start) >= 2 and start[:2].isdigit() else None
        wx = weather.get((stn, "%s %02d" % (g["date"], hour))) if (stn and hour is not None) else None

        rows.append([
            g["season"], g["date"], g["dow"], g["date"][5:7], start,
            g["home"], g["away"], g["stadium"],
            meta.get("home_teams", ""), meta.get("is_dome", ""), meta.get("is_secondary", ""),
            crowd, cap or "", occupancy, sold_out,
            stand.get("home_rank", ""), stand.get("home_wpct", ""), stand.get("home_gb", ""),
            stand.get("home_streak", ""), stand.get("home_last10", ""),
            stand.get("away_rank", ""), stand.get("away_wpct", ""), stand.get("away_gb", ""),
            stand.get("away_streak", ""), stand.get("away_last10", ""),
            stand.get("gb_from_playoff_line", ""), stand.get("season_progress", ""),
            (wx or {}).get("temp", ""), (wx or {}).get("rain", ""),
            game_rain(weather, stn, g["date"], hour) if (stn and hour is not None) else "",
            (wx or {}).get("rain_day", ""),
            (wx or {}).get("humid", ""), (wx or {}).get("wind", ""), (wx or {}).get("cloud", ""),
        ])

    rows.sort(key=lambda r: (r[1], r[7], r[5]))
    path = kbo.data_path("dataset.csv")
    n = kbo.write_csv(path, HEADER, rows)
    kbo.report("데이터셋", path, n)

    temp_at = HEADER.index("temp")
    filled = sum(1 for r in rows if r[temp_at] != "")
    print("  일정 미매칭 %d경기 / 날씨 있는 경기 %d (%d%%)"
          % (unmatched_schedule, filled, round(filled / len(rows) * 100) if rows else 0))


if __name__ == "__main__":
    main()
