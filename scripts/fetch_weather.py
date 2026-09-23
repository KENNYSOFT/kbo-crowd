"""구장 인근 기상 관측값을 받아 data/weather.csv 로 쓴다.

출처: 기상청 API 허브 https://apihub.kma.go.kr (ASOS 시간자료 kma_sfctm3)
인증키가 필요하다. apihub.kma.go.kr 에서 가입하면 바로 발급되고, 환경변수
KMA_API_KEY 로 넘긴다. 키가 없으면 아무것도 하지 않고 조용히 끝낸다.
날씨 없이도 나머지 수집은 굴러가야 하기 때문이다.

응답은 공백으로 구분된 텍스트다. 주석 줄에 컬럼명이 있지만 두 줄로 쪼개져
있어서(GST_WD 가 첫 줄 GST 와 둘째 줄 WD 로 갈린다) 헤더를 읽어 위치를 잡을
수 없다. 그래서 문서에 정의된 컬럼 순서를 코드에 두고 인덱스로 읽는다.

한 가지 조심할 것이 WW(25번, 현재일기)다. 22자 문자열이라 값에 공백이 들어갈
수 있고 그러면 그 뒤가 통째로 밀린다. 그래서 WW 앞까지만 split 을 신뢰하고,
필드 수가 정확히 맞을 때만 뒤쪽(운량 등)을 쓴다.

받는 범위는 구장이 있는 지점 × 경기가 있던 달이다. 전국 전체를 받을 이유가 없다.
서버가 간헐적으로 504 를 내는데 기간 길이와 무관하므로 재시도로 넘긴다.
"""

import argparse
import datetime as dt
import os
import sys
import time
from collections import defaultdict

import kbo

API = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"

# 지상관측 시간자료의 컬럼 순서. kma_sfctm2 와 kma_sfctm3 이 같다.
FIELDS = [
    "TM", "STN", "WD", "WS", "GST_WD", "GST_WS", "GST_TM", "PA", "PS", "PT", "PR",
    "TA", "TD", "HM", "PV", "RN", "RN_DAY", "RN_JUN", "RN_INT", "SD_HR3", "SD_DAY",
    "SD_TOT", "WC", "WP", "WW", "CA_TOT", "CA_MID", "CH_MIN", "CT", "CT_TOP",
    "CT_MID", "CT_LOW", "VS", "SS", "SI", "ST_GD", "TS", "TE_005", "TE_01",
    "TE_02", "TE_03", "ST_SEA", "WH", "BF", "IR", "IX",
]
SAFE = FIELDS.index("WW")  # 여기까지는 split 이 안전하다

# 쓰고 싶은 항목과 CSV 컬럼명. 오른쪽 값이 True 면 결측을 0 으로 본다.
WANTED = [
    ("TA", "temp", False),       # 기온 (도)
    ("RN", "rain", True),        # 그 시각 강수량 (mm)
    ("RN_DAY", "rain_day", True),  # 그날 누적 강수량 (mm)
    ("HM", "humid", False),      # 상대습도 (%)
    ("WS", "wind", False),       # 풍속 (m/s)
    ("CA_TOT", "cloud", False),  # 전운량 (1/10)
]
HEADER = ["stn", "tm", "temp", "rain", "rain_day", "humid", "wind", "cloud"]

MISSING = {"-9", "-9.0", "-99", "-99.0", "-999", "-999.0", "-9999", ""}


def parse_response(text):
    """고정된 컬럼 순서로 값을 뽑는다.

    강수는 결측을 0 으로 본다. 기상청은 비가 오지 않은 시각의 RN 을 -9.0 으로
    주는데, 이것을 결측으로 버리면 '비가 안 왔다'는 정보가 통째로 사라진다.
    비가 그친 뒤에도 RN 은 -9.0 이 되지만 RN_DAY 는 그날 누적을 유지하므로,
    둘을 함께 두면 '그날 비가 왔는지'와 '그 시각에 왔는지'를 갈라 볼 수 있다.

    다만 11월부터 이듬해 3월까지는 RN 이 1시간이 아니라 3시간 누적이라
    (3, 6, 9시 등) 개막 시기 경기는 시간대 강수가 어긋난다. 그 구간은
    rain 보다 rain_day 를 믿는 편이 낫다.
    """
    rows = []
    saw_data = False
    for line in text.splitlines():
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < SAFE:
            continue
        record = dict(zip(FIELDS, parts))
        if len(parts) != len(FIELDS):
            # WW 에 공백이 섞여 뒤쪽이 밀렸다. 앞쪽만 신뢰한다.
            for name in FIELDS[SAFE:]:
                record.pop(name, None)
        tm = record.get("TM", "")
        if len(tm) < 12 or not tm.isdigit():
            continue
        saw_data = True
        stamp = "%s-%s-%s %s:%s" % (tm[0:4], tm[4:6], tm[6:8], tm[8:10], tm[10:12])
        out = [record.get("STN", ""), stamp]
        for src, _, zero_when_missing in WANTED:
            value = record.get(src, "")
            if value in MISSING:
                out.append("0" if zero_when_missing else "")
            else:
                out.append(value)
        rows.append(out)
    if not saw_data:
        raise RuntimeError(
            "응답에 관측 행이 없다. 인증키나 활용신청, API 형식을 확인할 것.\n" + text[:300]
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


def have_enough(rows, stn, start, end, ratio=0.95):
    """그 구간이 이미 충분히 채워져 있는지.

    한 시간에 한 행이므로 기대 행 수는 일수 곱하기 24 다. 관측이 빠지는 시각도
    있어 전부를 요구하지는 않는다.
    """
    expected = ((end - start).days + 1) * 24
    prefix_start, prefix_end = start.isoformat(), end.isoformat()
    got = sum(
        1
        for (s, tm) in rows
        if s == stn and prefix_start <= tm[:10] <= prefix_end
    )
    return got >= expected * ratio


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
    ap.add_argument("--refetch", action="store_true",
                    help="이미 채워진 구간도 다시 받는다. 기본은 빠진 곳만 메운다")
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

    today = kbo.today_kst()
    since = None
    if args.recent_days:
        since = today - dt.timedelta(days=args.recent_days)
    elif args.since:
        since = dt.date.fromisoformat(args.since)

    # 지점별로 필요한 날짜를 모은다. 일정에는 아직 열리지 않은 경기도 있는데
    # 그 날의 관측은 존재하지 않으므로 요청하지 않는다.
    need = defaultdict(set)
    for r in schedule:
        stn = stn_of.get(r["stadium"])
        if not stn:
            continue
        day = dt.date.fromisoformat(r["date"])
        if day > today or (since and day < since):
            continue
        need[stn].add(day)

    existing = {(r["stn"], r["tm"]): r for r in kbo.read_csv(kbo.data_path("weather.csv"))}
    merged = dict(existing)
    ok = failed = skipped = 0
    first_error = None

    for stn in sorted(need):
        for start, end in month_spans(need[stn]):
            # 이미 채워진 구간은 건너뛴다. 서버가 간헐적으로 504 를 내서 구멍이
            # 남는데, 그때 이 스크립트를 다시 돌리면 빠진 곳만 메우게 된다.
            if not args.refetch and have_enough(merged, stn, start, end):
                skipped += 1
                continue
            try:
                for row in fetch_range(stn, start, end, key):
                    merged[(row[0], row[1])] = dict(zip(HEADER, row))
                ok += 1
            except Exception as exc:
                failed += 1
                if first_error is None:
                    first_error = str(exc)
                print("  %s %s~%s 실패: %s" % (stn, start, end, exc), file=sys.stderr)
            time.sleep(0.3)
        print("  지점 %s: 누적 %d행" % (stn, sum(1 for k in merged if k[0] == stn)))

    if skipped:
        print("  이미 채워져 있어 건너뛴 구간 %d개 (--refetch 로 강제)" % skipped)

    # 전부 실패했는데 0행으로 조용히 끝나면 성공한 것처럼 보인다.
    # 건너뛴 것이 있으면 기존 데이터가 남아 있으므로 치명적이지 않다.
    if ok == 0 and failed and not skipped:
        hint = ""
        # urllib 은 403 을 예외로 만들어 본문을 못 읽으므로 코드로도 판정한다.
        if first_error and ("활용신청" in first_error or "403" in first_error):
            hint = ("\n  인증키는 유효하지만 이 API 에 활용신청이 안 돼 있다.\n"
                    "  apihub.kma.go.kr 에서 지상관측 ASOS 시간자료를 신청할 것.")
        raise SystemExit("날씨 요청 %d건이 모두 실패했다.%s" % (failed, hint))

    rows = [[r.get(h, "") for h in HEADER] for r in merged.values()]
    rows.sort(key=lambda r: (str(r[0]), str(r[1])))
    path = kbo.data_path("weather.csv")
    n = kbo.write_csv(path, HEADER, rows)
    kbo.report("날씨", path, n)


if __name__ == "__main__":
    main()
