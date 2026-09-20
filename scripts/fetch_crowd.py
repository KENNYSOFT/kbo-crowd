"""KBO 기록실의 경기별 관중 수를 받아 data/games.csv 로 쓴다.

출처: https://www.koreabaseball.com/Record/Crowd/GraphDaily.aspx

이 페이지는 ASP.NET WebForms 라 시즌 선택이 포스트백이다. 대신 페이지네이션이
없어서 요청 한 번에 그 시즌 전체(720경기)가 온다. 그래서 증분 수집 대신 매번
전 시즌을 다시 받아 통째로 덮어쓴다. 요청이 시즌 수만큼(현재 6회)이라 부담이
없고, KBO 가 과거 수치를 정정했을 때 그것도 자동으로 따라온다.

경기 단위 데이터는 2021년부터만 나온다. 그 이전 시즌을 넣으면 빈 결과가 온다.
"""

import argparse
import html
import re
import sys
import time

import kbo

URL = kbo.BASE + "/Record/Crowd/GraphDaily.aspx"
PREFIX = "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$"
TOKENS = ("__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTVALIDATION")

ROW_RE = re.compile(r'<tr class="order">(.*?)</tr>', re.S)
CELL_RE = re.compile(r"<td>(.*?)</td>", re.S)
SEASON_OPT_RE = re.compile(
    r'id="cphContents_cphContents_cphContents_ddlSeason".*?</select>', re.S
)


def read_tokens(page):
    out = {}
    for name in TOKENS:
        m = re.search(r'id="%s" value="([^"]*)"' % name, page)
        if not m:
            raise RuntimeError("포스트백 토큰 %s 를 찾지 못했다. 페이지 구조가 바뀌었을 수 있다." % name)
        out[name] = html.unescape(m.group(1))
    return out


def read_seasons(page):
    """시즌 드롭다운에서 선택 가능한 연도를 읽는다.

    하드코딩하지 않는 이유는 해가 바뀌면 KBO 가 옵션을 늘리기 때문이다.
    맨 위 '시즌별' 항목은 현재 연도와 같은 값을 쓰므로 집합으로 중복을 없앤다.
    """
    block = SEASON_OPT_RE.search(page)
    if not block:
        raise RuntimeError("시즌 드롭다운을 찾지 못했다.")
    years = {v for v in re.findall(r'value="(\d{4})"', block.group(0))}
    return sorted(years)


def parse_rows(page, season):
    rows = []
    for chunk in ROW_RE.findall(page):
        cells = [c.strip() for c in CELL_RE.findall(chunk)]
        if len(cells) != 6:
            continue
        date, dow, home, away, stadium, crowd = cells
        rows.append(
            [
                season,
                date.replace("/", "-"),
                dow,
                home,
                away,
                stadium,
                int(crowd.replace(",", "")),
            ]
        )
    return rows


def fetch_season(season, tokens):
    form = dict(tokens)
    form.update(
        {
            PREFIX + "ddlSeason": season,
            PREFIX + "ddlMonth": "0",
            PREFIX + "ddlTeam": "",
            PREFIX + "ddlHomeAway": "",
            PREFIX + "ddlStadium": "",
            PREFIX + "ddlDayOfWeek": "0",
            PREFIX + "btnSearch": "검색",
        }
    )
    page = kbo.http(URL, data=form)
    return parse_rows(page, season), read_tokens(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs="*", help="받을 시즌. 비우면 페이지가 제공하는 전체")
    args = ap.parse_args()

    page = kbo.http(URL)
    tokens = read_tokens(page)
    seasons = args.seasons or read_seasons(page)

    all_rows = []
    for season in seasons:
        rows, tokens = fetch_season(season, tokens)
        if not rows:
            print("  %s: 0행 (경기 단위 데이터는 2021년부터 제공된다)" % season, file=sys.stderr)
        else:
            print("  %s: %d경기" % (season, len(rows)))
        all_rows.extend(rows)
        time.sleep(0.7)

    if not all_rows:
        raise SystemExit("수집된 경기가 없다. 페이지 구조 변경을 의심할 것.")

    # 정렬을 고정해야 매일 덮어써도 diff 에 바뀐 줄만 남는다.
    all_rows.sort(key=lambda r: (r[1], r[5], r[3]))
    path = kbo.data_path("games.csv")
    n = kbo.write_csv(
        path, ["season", "date", "dow", "home", "away", "stadium", "crowd"], all_rows
    )
    kbo.report("관중", path, n)


if __name__ == "__main__":
    main()
