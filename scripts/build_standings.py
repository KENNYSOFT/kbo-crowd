"""경기 결과에서 각 경기 시점의 순위와 연승을 계산해 data/standings.csv 로 쓴다.

KBO 의 순위 페이지를 날짜마다 긁는 대신 직접 계산한다. 요청이 시즌당 170번쯤
줄어들기도 하지만, 더 중요한 이유는 기준 시점을 내가 통제할 수 있어서다.

여기서 만드는 값은 전부 '그 경기가 시작되기 전'의 상태다. 관중을 예측할 때
쓸 수 있는 정보는 표를 사는 시점까지의 것뿐이라, 같은 날 경기 결과가 섞여
들어가면 그대로 누수가 된다. 그래서 하루치 경기의 피처를 모두 만든 다음에
그날 결과를 반영한다.

승률은 KBO 방식으로 무승부를 제외하고 계산한다(승 / (승+패)).
연속 기록도 KBO 방식으로 무승부가 연승이나 연패를 끊지 않는다.
"""

import argparse
from collections import defaultdict

import kbo

HEADER = [
    "date",
    "stadium",
    "home",
    "away",
    "home_rank",
    "home_wpct",
    "home_gb",
    "home_streak",
    "home_last10",
    "home_games",
    "away_rank",
    "away_wpct",
    "away_gb",
    "away_streak",
    "away_last10",
    "away_games",
    "gb_from_playoff_line",
    "season_progress",
]

# 정규시즌 경기 수. 10개 구단이 144경기씩 치른다. 진행도의 분모를 결과가 난 경기
# 수로 두면 진행 중인 시즌에서는 앞으로 열릴 경기가 5월이든 10월이든 1.0 이 된다.
SEASON_GAMES = 10 * 144 // 2


class Team:
    __slots__ = ("w", "l", "d", "streak", "recent")

    def __init__(self):
        self.w = self.l = self.d = 0
        self.streak = 0  # 양수는 연승, 음수는 연패
        self.recent = []  # 최근 결과, 오래된 것부터

    @property
    def games(self):
        return self.w + self.l + self.d

    @property
    def wpct(self):
        decided = self.w + self.l
        return self.w / decided if decided else 0.0

    def last10(self):
        window = self.recent[-10:]
        return sum(1 for x in window if x == "W")

    def apply(self, result):
        if result == "W":
            self.w += 1
            self.streak = self.streak + 1 if self.streak > 0 else 1
        elif result == "L":
            self.l += 1
            self.streak = self.streak - 1 if self.streak < 0 else -1
        else:
            self.d += 1  # 무승부는 연속을 유지한다
        self.recent.append(result)


def rank_table(teams):
    """승률 내림차순 순위와 1위 대비 게임차를 낸다."""
    order = sorted(teams.items(), key=lambda kv: (-kv[1].wpct, -kv[1].w))
    if not order:
        return {}, {}
    top = order[0][1]
    ranks, gaps = {}, {}
    for i, (name, t) in enumerate(order, start=1):
        ranks[name] = i
        gaps[name] = ((top.w - t.w) + (t.l - top.l)) / 2
    return ranks, gaps


def playoff_line_gb(teams, ranks, name):
    """5위(가을야구 진출선)와의 게임차. 5위 안이면 음수가 되어 여유를 뜻한다."""
    order = sorted(teams.items(), key=lambda kv: (-kv[1].wpct, -kv[1].w))
    if len(order) < 5:
        return ""
    fifth = order[4][1]
    me = teams[name]
    return round(((fifth.w - me.w) + (me.l - fifth.l)) / 2, 1)


def has_result(row):
    return row["home_score"] != "" and row["away_score"] != ""


def build(schedule_rows, season):
    """그 시즌 모든 경기의 '경기 직전' 상태를 낸다.

    아직 열리지 않은 경기도 포함한다. 오늘 경기의 관중을 예측하려면 오늘의
    순위가 필요한데, 결과가 있는 경기만 다루면 예정 경기에는 아무것도 붙지
    않는다. 상태를 갱신하는 것은 결과가 나온 경기뿐이다.
    """
    games = [r for r in schedule_rows if r["season"] == season]
    if not games:
        return []

    by_date = defaultdict(list)
    for r in games:
        by_date[r["date"]].append(r)

    teams = defaultdict(Team)
    played = 0
    out = []

    for date in sorted(by_date):
        todays = by_date[date]
        ranks, gaps = rank_table(teams)

        for r in todays:
            home, away = r["home"], r["away"]
            ht, at = teams[home], teams[away]
            out.append(
                [
                    date,
                    r["stadium"],
                    home,
                    away,
                    ranks.get(home, ""),
                    round(ht.wpct, 3) if ht.games else "",
                    gaps.get(home, ""),
                    ht.streak,
                    ht.last10() if ht.games else "",
                    ht.games,
                    ranks.get(away, ""),
                    round(at.wpct, 3) if at.games else "",
                    gaps.get(away, ""),
                    at.streak,
                    at.last10() if at.games else "",
                    at.games,
                    playoff_line_gb(teams, ranks, home) if ht.games else "",
                    round(played / SEASON_GAMES, 3),
                ]
            )

        for r in todays:
            if not has_result(r):
                continue  # 취소됐거나 아직 열리지 않은 경기는 순위를 움직이지 않는다
            hs, aws = int(r["home_score"]), int(r["away_score"])
            if hs > aws:
                teams[r["home"]].apply("W")
                teams[r["away"]].apply("L")
            elif hs < aws:
                teams[r["home"]].apply("L")
                teams[r["away"]].apply("W")
            else:
                teams[r["home"]].apply("D")
                teams[r["away"]].apply("D")
            played += 1

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs="*", help="계산할 시즌. 비우면 일정에 있는 전체")
    args = ap.parse_args()

    schedule = kbo.read_csv(kbo.data_path("schedule.csv"))
    if not schedule:
        raise SystemExit("data/schedule.csv 가 없다. 먼저 fetch_schedule.py 를 실행할 것.")

    seasons = args.seasons or sorted({r["season"] for r in schedule})
    rows = []
    for season in seasons:
        built = build(schedule, season)
        print("  %s: %d경기" % (season, len(built)))
        rows.extend(built)

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    path = kbo.data_path("standings.csv")
    n = kbo.write_csv(path, HEADER, rows)
    kbo.report("순위", path, n)


if __name__ == "__main__":
    main()
