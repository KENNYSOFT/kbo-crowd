"""앞으로 열릴 경기의 기상 예보를 받아 data/forecast.csv 로 쓴다.

출처: 기상청 API 허브 단기예보조회(VilageFcstInfoService_2.0/getVilageFcst)
인증키가 필요하고, 그 API 에 활용신청이 따로 되어 있어야 한다.

fetch_weather.py 가 받는 것은 지나간 관측이라 아직 열리지 않은 경기에는 쓸 수
없다. 오늘이나 내일 경기의 매진 확률을 내려면 예보가 있어야 한다.

단기예보는 격자로 조회한다. 위경도를 그대로 넘길 수 없어서 구장 좌표를
기상청 격자(nx, ny)로 변환하는데, 이 변환은 공개된 Lambert Conformal Conic
공식이라 별도 API 없이 계산된다(서울 종로 60,127 과 사직 98,76 으로 확인).

예보는 발표할 때마다 바뀌므로 이 파일은 매번 통째로 덮어쓴다. 과거 예보를
쌓아 두지 않는 것은 지금 목적이 예측이지 예보 정확도 평가가 아니기 때문이다.
"""

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import time

import kbo

API = "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst"

# 단기예보 발표 시각(KST). 발표 뒤 10분쯤 지나야 조회된다.
BASE_HOURS = [2, 5, 8, 11, 14, 17, 20, 23]
PUBLISH_DELAY_MIN = 15

HEADER = ["stadium", "date", "hour", "temp", "rain_mm", "rain_prob", "rain_type",
          "humid", "wind", "sky", "base_date", "base_time"]

# 하늘상태 코드를 관측의 전운량(0~10) 스케일로 옮긴다. 모델이 두 자료를 같은
# 컬럼으로 받으므로 눈금을 맞춰야 한다. 기상청 정의의 중간값을 쓴다.
SKY_TO_CLOUD = {"1": 2.5, "3": 7.0, "4": 9.5}


def to_grid(lat, lon):
    """위경도를 기상청 동네예보 격자로 바꾼다."""
    RE, GRID = 6371.00877, 5.0
    SLAT1, SLAT2, OLON, OLAT, XO, YO = 30.0, 60.0, 126.0, 38.0, 43, 136
    D = math.pi / 180.0
    re_ = RE / GRID
    s1, s2, olon, olat = SLAT1 * D, SLAT2 * D, OLON * D, OLAT * D
    sn = math.log(math.cos(s1) / math.cos(s2)) / math.log(
        math.tan(math.pi * 0.25 + s2 * 0.5) / math.tan(math.pi * 0.25 + s1 * 0.5))
    sf = (math.tan(math.pi * 0.25 + s1 * 0.5) ** sn) * math.cos(s1) / sn
    ro = re_ * sf / (math.tan(math.pi * 0.25 + olat * 0.5) ** sn)
    ra = re_ * sf / (math.tan(math.pi * 0.25 + lat * D * 0.5) ** sn)
    theta = lon * D - olon
    if theta > math.pi:
        theta -= 2 * math.pi
    if theta < -math.pi:
        theta += 2 * math.pi
    theta *= sn
    return int(ra * math.sin(theta) + XO + 0.5), int(ro - ra * math.cos(theta) + YO + 0.5)


def latest_base(now):
    """그 시각에 조회 가능한 가장 최근 발표를 고른다."""
    t = now - dt.timedelta(minutes=PUBLISH_DELAY_MIN)
    for hour in reversed(BASE_HOURS):
        if t.hour >= hour:
            return t.strftime("%Y%m%d"), "%02d00" % hour
    yesterday = t - dt.timedelta(days=1)
    return yesterday.strftime("%Y%m%d"), "2300"


def parse_pcp(value):
    """강수량 표기를 mm 숫자로 바꾼다.

    기상청은 이 값을 숫자가 아니라 범주 문자열로 준다. 비가 없으면 '강수없음'
    이나 '0', 있으면 '1.0mm', 구형 표기로는 '1.0~29.9mm' 나 '50.0mm 이상' 이
    온다. 범위는 아래값을, 이상은 그 값을 쓴다. 모르는 표기는 0 이 아니라
    빈 값으로 두어야 '비가 안 왔다'와 '읽지 못했다'가 섞이지 않는다.
    """
    text = (value or "").strip()
    if not text or text in ("강수없음", "적설없음", "-"):
        return 0.0
    cleaned = text.replace("mm", "").replace("이상", "").replace("미만", "").strip()
    if "~" in cleaned:
        cleaned = cleaned.split("~")[0]
    try:
        return float(cleaned)
    except ValueError:
        return None


def fetch_grid(nx, ny, base_date, base_time, key):
    url = ("%s?pageNo=1&numOfRows=1200&dataType=JSON&base_date=%s&base_time=%s"
           "&nx=%d&ny=%d&authKey=%s" % (API, base_date, base_time, nx, ny, key))
    payload = json.loads(kbo.http(url))
    body = payload.get("response", {}).get("body", {})
    items = body.get("items", {}).get("item", [])
    if not items:
        head = payload.get("response", {}).get("header", {})
        raise RuntimeError("예보가 비었다: %s" % head.get("resultMsg", payload)[:200])

    merged = {}
    for item in items:
        key_ = (item["fcstDate"], item["fcstTime"])
        merged.setdefault(key_, {})[item["category"]] = item["fcstValue"]
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=3, help="오늘부터 며칠치 경기를 볼지")
    args = ap.parse_args()

    key = os.environ.get("KMA_API_KEY", "").strip()
    if not key:
        print("KMA_API_KEY 가 없어 예보 수집을 건너뛴다.", file=sys.stderr)
        return

    stadiums = kbo.read_csv(kbo.data_path("stadiums.csv"))
    schedule = kbo.read_csv(kbo.data_path("schedule.csv"))
    if not schedule:
        raise SystemExit("data/schedule.csv 가 없다. 먼저 fetch_schedule.py 를 실행할 것.")

    today = dt.date.today()
    until = today + dt.timedelta(days=args.days)
    upcoming = {r["stadium"] for r in schedule
                if today.isoformat() <= r["date"] <= until.isoformat()}
    if not upcoming:
        print("  %s 까지 예정 경기가 없어 받을 예보가 없다." % until)
        kbo.write_csv(kbo.data_path("forecast.csv"), HEADER, [])
        return

    base_date, base_time = latest_base(dt.datetime.now())
    print("  발표 기준 %s %s / 대상 구장 %d곳" % (base_date, base_time, len(upcoming)))

    rows = []
    for meta in stadiums:
        name = meta["stadium"]
        if name not in upcoming or not meta.get("lat"):
            continue
        nx, ny = to_grid(float(meta["lat"]), float(meta["lon"]))
        try:
            grid = fetch_grid(nx, ny, base_date, base_time, key)
        except Exception as exc:
            print("  %s (nx=%d ny=%d) 실패: %s" % (name, nx, ny, exc), file=sys.stderr)
            continue
        for (fdate, ftime), values in sorted(grid.items()):
            pcp = parse_pcp(values.get("PCP"))
            rows.append([
                name,
                "%s-%s-%s" % (fdate[0:4], fdate[4:6], fdate[6:8]),
                ftime[:2],
                values.get("TMP", ""),
                "" if pcp is None else pcp,
                values.get("POP", ""),
                values.get("PTY", ""),
                values.get("REH", ""),
                values.get("WSD", ""),
                SKY_TO_CLOUD.get(values.get("SKY", ""), ""),
                base_date,
                base_time,
            ])
        print("  %s: nx=%d ny=%d, %d시각" % (name, nx, ny, len(grid)))
        time.sleep(0.3)

    if not rows:
        raise SystemExit("받은 예보가 없다. 인증키와 활용신청을 확인할 것.")

    rows.sort(key=lambda r: (str(r[1]), str(r[2]), str(r[0])))
    path = kbo.data_path("forecast.csv")
    n = kbo.write_csv(path, HEADER, rows)
    kbo.report("예보", path, n)


if __name__ == "__main__":
    main()
