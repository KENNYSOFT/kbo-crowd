"""구장 인근 기상 관측값을 받아 data/weather.csv 로 쓴다.

출처: 기상청 API 허브 https://apihub.kma.go.kr (ASOS 시간자료 kma_sfctm3)
인증키가 필요하다. apihub.kma.go.kr 에서 가입하면 바로 발급되고, 환경변수
KMA_API_KEY 로 넘긴다. 키가 없으면 아무것도 하지 않고 조용히 끝낸다.
날씨 없이도 나머지 수집은 굴러가야 하기 때문이다.

응답은 공백으로 구분된 고정폭 텍스트다. 컬럼 순서를 코드에 박아 두면 기상청이
항목을 하나 끼워 넣는 순간 값이 통째로 밀리므로, 주석 줄의 컬럼명을 읽어
위치를 그때그때 찾는다.

받는 범위는 구장이 있는 지점 × 경기가 있던 날이다. 전국 전체를 받을 이유가 없다.
"""

import argparse
import datetime as dt
import os
import sys
import time
from collections import defaultdict

import kbo

API = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"

# 쓰고 싶은 항목과 CSV 컬럼명. 기상청 표기는 왼쪽.
WANTED = [
    ("TA", "temp"),       # 기온 (도)
    ("RN", "rain"),       # 강수량 (mm, 매시간)
    ("HM", "humid"),      # 상대습도 (%)
    ("WS", "wind"),       # 풍속 (m/s)
    ("CA_TOT", "cloud"),  # 전운량 (1/10)
]
HEADER = ["stn", "tm", "temp", "rain", "humid", "wind", "cloud"]

MISSING = {"-9", "-9.0", "-99", "-99.0", "-999", "-999.0", "-9999", ""}


def parse_response(text):
    """헤더 주석에서 컬럼 위치를 찾아 값을 뽑는다."""
    names = None
    rows = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("#"):
            # 컬럼명이 들어 있는 주석 줄을 찾는다. TM 과 STN 이 함께 있으면 그 줄이다.
            candidate = line.lstrip("#").split()
            if "TM" in candidate and "STN" in candidate:
                names = candidate
            continue
        if names is None:
            continue
        parts = line.split()
        if len(parts) < len(names):
            continue
        record = dict(zip(names, parts))
        tm = record.get("TM", "")
        if len(tm) < 12:
            continue
        stamp = "%s-%s-%s %s:%s" % (tm[0:4], tm[4:6], tm[6:8], tm[8:10], tm[10:12])
        out = [record.get("STN", ""), stamp]
        for src, _ in WANTED:
            value = record.get(src, "")
            out.append("" if value in MISSING else value)
        rows.append(out)
    if names is None:
        raise RuntimeError(
            "응답에서 컬럼명 줄을 찾지 못했다. 인증키가 잘못됐거나 API 형식이 바뀌었다.\n"
            + text[:300]
        )
    return rows


def fetch_range(stn, start, end, key):
    url = "%s?tm1=%s0000&tm2=%s2300&stn=%s&help=1&authKey=%s" % (
        API,
        start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d"),
        stn,
        key,
    )
    return parse_response(kbo.http(url))


def month_spans(dates):
    """날짜 집합을 달 단위 구간으로 묶는다. 요청 수를 줄이려는 것."""
    by_month = defaultdict(list)
    for d in dates:
        by_month[(d.year, d.month)].append(d)
    for (year, month), days in sorted(by_month.items()):
        yield min(days), max(days)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", help="이 날짜(YYYY-MM-DD) 이후 경기만. 비우면 전체")
    ap.add_argument("--recent-days", type=int, help="최근 N일치만 받는다. 일일 수집용")
    args = ap.parse_args()

    key = os.environ.get("KMA_API_KEY", "").strip()
    if not key:
        print("KMA_API_KEY 가 없어 날씨 수집을 건너뛴다.", file=sys.stderr)
        print("  키 발급: https://apihub.kma.go.kr 가입 후 마이페이지", file=sys.stderr)
        return

    stadiums = kbo.read_csv(kbo.data_path("stadiums.csv"))
    stn_of = {r["stadium"]: r["asos_stn"] for r in stadiums if r["asos_stn"]}
    schedule = kbo.read_csv(kbo.data_path("schedule.csv"))
    if not schedule:
        raise SystemExit("data/schedule.csv 가 없다. 먼저 fetch_schedule.py 를 실행할 것.")

    since = None
    if args.recent_days:
        since = dt.date.today() - dt.timedelta(days=args.recent_days)
    elif args.since:
        since = dt.date.fromisoformat(args.since)

    # 지점별로 필요한 날짜를 모은다.
    need = defaultdict(set)
    for r in schedule:
        stn = stn_of.get(r["stadium"])
        if not stn:
            continue
        day = dt.date.fromisoformat(r["date"])
        if since and day < since:
            continue
        need[stn].add(day)

    existing = {(r["stn"], r["tm"]): r for r in kbo.read_csv(kbo.data_path("weather.csv"))}
    merged = dict(existing)
    added = 0

    for stn in sorted(need):
        for start, end in month_spans(need[stn]):
            try:
                for row in fetch_range(stn, start, end, key):
                    merged[(row[0], row[1])] = dict(zip(HEADER, row))
                    added += 1
            except Exception as exc:
                print("  %s %s~%s 실패: %s" % (stn, start, end, exc), file=sys.stderr)
            time.sleep(0.3)
        print("  지점 %s: 누적 %d행" % (stn, sum(1 for k in merged if k[0] == stn)))

    rows = [[r.get(h, "") for h in HEADER] for r in merged.values()]
    rows.sort(key=lambda r: (str(r[0]), str(r[1])))
    path = kbo.data_path("weather.csv")
    n = kbo.write_csv(path, HEADER, rows)
    kbo.report("날씨", path, n)


if __name__ == "__main__":
    main()
