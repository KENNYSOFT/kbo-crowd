"""KBO 경기 일정과 결과를 받아 data/schedule.csv 로 쓴다.

출처: POST https://www.koreabaseball.com/ws/Schedule.asmx/GetScheduleList
일정 페이지가 쓰는 내부 API 다. jQuery 가 form 으로 보내므로 JSON 본문이 아니라
form 인코딩으로 요청해야 한다(JSON 으로 보내면 에러 페이지가 온다).

관중 CSV 에 없는 세 가지를 여기서 얻는다.
  - 경기 시작 시각: 날씨를 경기 시간대에 맞춰 붙이려면 필요하다
  - 스코어와 승패: 순위와 연승을 직접 계산하는 재료
  - 우천취소 여부: 취소된 경기는 관중 CSV 에 아예 없다
"""

import argparse
import datetime as dt
import json
import re
import sys
import time

import kbo

URL = kbo.BASE + "/ws/Schedule.asmx/GetScheduleList"
REGULAR_SERIES = "0,9,6"  # 정규시즌. 시범경기는 "1", 포스트시즌은 "3,4,5,7"
MONTHS = ["03", "04", "05", "06", "07", "08", "09", "10", "11"]

PLAY_RE = re.compile(r"<span>(.*?)</span><em>(.*?)</em><span>(.*?)</span>", re.S)
DAY_RE = re.compile(r"(\d{2})\.(\d{2})")
TAG_RE = re.compile(r"<[^>]+>")


def strip_tags(text):
    return TAG_RE.sub("", text or "").strip()


def parse_month(payload, season):
    """한 달치 응답을 경기 행으로 편다.

    첫 경기 행에만 날짜 셀이 있고 나머지는 rowspan 으로 묶여 있어서,
    날짜를 직전 값에서 이어받아야 한다.
    """
    rows = []
    current_date = None
    for entry in payload.get("rows", []):
        cells = entry.get("row", [])
        by_class = {}
        for cell in cells:
            cls = cell.get("Class")
            if cls and cls not in by_class:
                by_class[cls] = cell.get("Text", "")

        if "day" in by_class:
            m = DAY_RE.search(strip_tags(by_class["day"]))
            if m:
                current_date = "%s-%s-%s" % (season, m.group(1), m.group(2))
        if not current_date:
            continue

        play = by_class.get("play", "")
        m = PLAY_RE.search(play)
        if not m:
            continue
        away = strip_tags(m.group(1))
        middle = m.group(2)
        home = strip_tags(m.group(3))
        scores = re.findall(r"<span[^>]*>(\d+)</span>", middle)
        away_score = int(scores[0]) if len(scores) == 2 else ""
        home_score = int(scores[1]) if len(scores) == 2 else ""

        # 뒤쪽 셀은 순서대로 하이라이트, TV, 라디오, 구장, 비고 다.
        tail = [strip_tags(c.get("Text", "")) for c in cells]
        stadium = tail[-2] if len(tail) >= 2 else ""
        note = tail[-1] if tail else ""
        start = strip_tags(by_class.get("time", ""))

        rows.append(
            [
                season,
                current_date,
                start,
                home,
                away,
                stadium,
                home_score,
                away_score,
                note,
            ]
        )
    return rows


def fetch(season, month):
    body = kbo.http(
        URL,
        data={
            "leId": "1",
            "srIdList": REGULAR_SERIES,
            "seasonId": season,
            "gameMonth": month,
            "teamId": "",
        },
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "Referer": kbo.BASE + "/Schedule/Schedule.aspx",
        },
    )
    return json.loads(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs="*", help="받을 시즌. 비우면 2021부터 올해까지")
    ap.add_argument(
        "--current-only",
        action="store_true",
        help="올해만 받아 기존 CSV 에 덮어쓴다. 확정된 과거 결과는 다시 받지 않는다",
    )
    args = ap.parse_args()

    this_year = dt.date.today().year
    if args.seasons:
        seasons = args.seasons
    elif args.current_only:
        seasons = [str(this_year)]
    else:
        seasons = [str(y) for y in range(2021, this_year + 1)]

    fetched = []
    for season in seasons:
        count = 0
        for month in MONTHS:
            try:
                payload = fetch(season, month)
            except Exception as exc:  # 한 달이 비어도 나머지는 계속 받는다
                print("  %s-%s 실패: %s" % (season, month, exc), file=sys.stderr)
                continue
            rows = parse_month(payload, season)
            fetched.extend(rows)
            count += len(rows)
            time.sleep(0.4)
        print("  %s: %d경기" % (season, count))

    path = kbo.data_path("schedule.csv")
    header = [
        "season",
        "date",
        "start",
        "home",
        "away",
        "stadium",
        "home_score",
        "away_score",
        "note",
    ]

    if args.current_only:
        kept = [
            [r[h] for h in header]
            for r in kbo.read_csv(path)
            if r["season"] not in set(seasons)
        ]
        fetched = kept + fetched

    if not fetched:
        raise SystemExit("수집된 일정이 없다. API 응답 구조 변경을 의심할 것.")

    fetched.sort(key=lambda r: (str(r[1]), str(r[5]), str(r[3])))
    n = kbo.write_csv(path, header, fetched)
    kbo.report("일정", path, n)


if __name__ == "__main__":
    main()
