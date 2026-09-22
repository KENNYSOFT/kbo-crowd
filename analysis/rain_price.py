"""비가 관중을 얼마나 깎는지 추정한다.

python analysis/rain_price.py

KMA_API_KEY 를 넣고 fetch_weather.py 를 먼저 돌려야 한다.

관중 수를 그냥 비교하면 답이 틀어진다. 비 오는 날은 주중에 몰리는 것도 아니고
특정 구장에 몰리는 것도 아니지만, 매진이 섞여 있기 때문이다. 표가 다 팔린
경기는 비가 와도 관중 수가 안 줄어든 것처럼 보인다. 이미 산 표는 비가 와도
그대로 집계되고, 애초에 그날 수요가 좌석보다 컸으니 조금 줄어도 상한에 걸린다.
그래서 여기서도 검열 회귀로 수요 쪽을 보고, 요일과 팀과 구장을 통제한 상태에서
비 계수만 읽는다.

돔구장을 따로 떼어 보는 것이 이 분석의 대조군이다. 고척은 비가 와도 경기가
열리므로 '비 때문에 관중이 주는' 경로가 경기 진행이 아니라 오가는 길에만 남는다.

그런데 그 계수만으로는 비의 값을 다 말할 수 없다. 비는 두 경로로 작동하는데
관중 기록에는 한쪽만 남기 때문이다. 비가 오면 (1) 경기가 아예 취소되고
(2) 열린 경기에는 덜 온다. 취소된 경기는 관중 기록에 없으므로 dataset.csv 만
보면 (2)만 재게 되고, 그마저 남은 표본이 약한 비로 쏠려 거의 0 으로 나온다.
비가 온 편성 357경기 중 217경기, 열에 여섯이 취소되어 사라진다.

그래서 일정에서 출발하는 표를 함께 낸다. 취소를 관중 0 으로 놓고 편성 한
경기당 기대 점유율을 내면, 표를 사려던 사람이 실제로 겪는 크기가 나온다.
그 값은 강수 10mm 이상에서 맑은 날의 1할 수준까지 떨어진다.

3월 경기는 기본으로 뺀다. 기상청 시간자료는 11월부터 이듬해 3월까지 강수를
1시간이 아니라 3시간 누적으로 주기 때문에, 경기 시간대 강수를 그대로 쓰면
개막 시기만 값이 부풀거나 비게 된다.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANDS = [-0.01, 0.0, 1.0, 3.0, 10.0, float("inf")]
BAND_LABELS = ["0mm", "0~1mm", "1~3mm", "3~10mm", "10mm+"]


def scheduled_with_rain(min_season, keep_march=False, span=3):
    """편성된 경기 전부에 그날 강수와 점유율을 붙인다.

    관중 데이터가 아니라 일정에서 출발하는 것이 핵심이다. 취소된 경기는 관중
    기록에 아예 없으므로 dataset.csv 로는 존재 자체를 볼 수 없는데, 일정에는
    남아 있고 관측소에는 그날 비가 얼마나 왔는지도 남아 있다.

    돔은 뺀다. 비로 취소되지 않아 이 비교의 대상이 아니다.
    """
    sch = pd.read_csv(os.path.join(ROOT, "data", "schedule.csv"), dtype={"season": str})
    sta = pd.read_csv(os.path.join(ROOT, "data", "stadiums.csv"))
    wx = pd.read_csv(os.path.join(ROOT, "data", "weather.csv"))

    station = dict(zip(sta["stadium"], sta["asos_stn"]))
    dome = dict(zip(sta["stadium"], sta["is_dome"]))
    wx["rain"] = pd.to_numeric(wx["rain"], errors="coerce")
    at = {(r.stn, r.tm[:10], r.tm[11:13]): r.rain for r in wx.itertuples()}

    def rain_at(stadium, date, start):
        stn, start = station.get(stadium), str(start)
        if stn is None or len(start) < 2 or not start[:2].isdigit():
            return np.nan
        hour, total, seen = int(start[:2]), 0.0, False
        for offset in range(span):
            v = at.get((stn, date, "%02d" % ((hour + offset) % 24)))
            if v is not None and pd.notna(v):
                total += float(v)
                seen = True
        return total if seen else np.nan

    sch = sch[sch["season"] >= str(min_season)].copy()
    sch["month"] = sch["date"].str[5:7]
    if not keep_march:
        sch = sch[sch["month"] != "03"]
    sch = sch[sch["stadium"].map(dome) != 1].copy()
    sch["cancelled"] = sch["note"].astype(str).str.contains("우천")
    sch["rain"] = [rain_at(r.stadium, r.date, r.start) for r in sch.itertuples()]

    played = model.load_dataset(min_season=min_season)
    occupancy = {(r.date, r.home, r.away): r.occupancy for r in played.itertuples()}
    sch["occupancy"] = [occupancy.get((r.date, r.home, r.away), np.nan)
                        for r in sch.itertuples()]

    # 그 경기가 평소 얼마나 팔리는 자리인지. 비와 무관하게 정해지는 값이라,
    # 취소가 강수뿐 아니라 흥행에도 갈리는지 보는 잣대로 쓴다.
    played = played.assign(dow_en=pd.to_datetime(played["date"]).dt.strftime("%a"))
    pull = played.groupby(["season", "home", "dow_en"])["occupancy"].mean()
    dow_en = pd.to_datetime(sch["date"]).dt.strftime("%a")
    sch["pull"] = [pull.get((s, h, d), np.nan)
                   for s, h, d in zip(sch["season"], sch["home"], dow_en)]
    return sch[sch["rain"].notna()].copy()


def selection_table(sch):
    """열린 경기가 흥행 쪽으로 걸러졌는지 본다.

    취소는 강수에만 갈리는 것이 아니다. 표가 많이 팔린 경기일수록 웬만하면
    열기 때문에, 비가 센 구간에 남은 경기는 원래 잘 팔리던 자리로 쏠린다.
    그래서 열린 경기의 점유율이 비가 셀수록 되레 높아 보인다.
    """
    sch = sch[sch["pull"].notna()].copy()
    sch["band"] = pd.cut(sch["rain"], bins=BANDS, labels=BAND_LABELS)

    print("=== 센 비에 열리는 경기는 원래 잘 팔리던 경기다 ===")
    print("  %-7s %6s %11s %11s %8s %10s"
          % ("강수", "편성", "편성 흥행", "열린 흥행", "선택 폭", "열린 점유"))
    for band, g in sch.groupby("band", observed=True):
        opened = g[~g["cancelled"] & g["occupancy"].notna()]
        if len(opened) < 3:
            continue
        booked, played_pull = g["pull"].mean(), opened["pull"].mean()
        print("  %-7s %6d %10.1f%% %10.1f%% %+7.1f%%p %9.1f%%"
              % (band, len(g), booked * 100, played_pull * 100,
                 (played_pull - booked) * 100, opened["occupancy"].mean() * 100))
    print("  편성 흥행은 그 경기가 평소 팔리는 정도, 열린 흥행은 그중 실제로 열린 것만")
    print("  추린 값이다. 선택 폭이 곧 취소가 표본을 기울인 크기다.")


def cancellation_table(sch):
    """비가 두 경로로 갈라져 작동하는 것을 한 표에 놓는다.

    지금까지 재던 것은 '열린 경기에 덜 오는' 쪽뿐인데, 비가 셀수록 경기 자체가
    열리지 않으므로 그쪽에 남는 표본은 약한 비로 쏠린다. 기대 점유율은 취소를
    관중 0 으로 보고 편성 한 경기당 평균을 낸 값이라, 표를 사려던 사람 입장에서
    '그날 경기를 볼 수 있었는가' 까지 담는다.
    """
    sch = sch.copy()
    sch["band"] = pd.cut(sch["rain"], bins=BANDS, labels=BAND_LABELS)

    print("=== 비는 두 경로로 작동한다 (야외, 3월 제외) ===")
    print("  %-7s %6s %6s %7s %6s %10s %10s %9s"
          % ("강수", "편성", "취소", "취소율", "열림", "열린 점유", "기대 점유", "기대 변화"))
    base = None
    for band, g in sch.groupby("band", observed=True):
        opened = g[~g["cancelled"] & g["occupancy"].notna()]
        if not len(opened):
            continue
        rate, occ = g["cancelled"].mean(), opened["occupancy"].mean()
        expected = (1 - rate) * occ
        if base is None:
            base = expected
        print("  %-7s %6d %6d %6.1f%% %6d %9.1f%% %9.1f%% %8.1f%%"
              % (band, len(g), g["cancelled"].sum(), rate * 100, len(opened),
                 occ * 100, expected * 100, (expected / base - 1) * 100))

    wet = sch[sch["rain"] > 0]
    print()
    print("  비가 온 편성 %d경기 중 %d경기(%.0f%%)가 취소되어 관중 기록에 없다."
          % (len(wet), wet["cancelled"].sum(), wet["cancelled"].mean() * 100))
    print("  열린 점유율이 센 비에서 되레 높은 것은 표본이 얇아서이기도 하고,")
    print("  표가 많이 팔린 경기일수록 웬만하면 열기 때문이기도 하다.")


def rain_effect(df, label):
    """그 부분집합에서 비 계수를 배수로 낸다.

    구장이 하나뿐인 부분집합(돔)에서는 팀과 구장 더미가 사실상 상수가 되어
    설계행렬이 특이해지고 최적화가 수렴하지 못한다. 그래서 왜 값을 못 냈는지
    사유를 함께 돌려준다. 조용히 '표본 부족'으로 뭉뚱그리면 실제로는 표본이
    충분한데 모델 쪽 문제인 경우를 놓친다.
    """
    wet = (pd.to_numeric(df["rain_game"], errors="coerce").fillna(0) > 0)
    if len(df) < 150:
        return label, len(df), None, None, int(wet.sum()), "경기 부족"
    if wet.sum() < 15 or (~wet).sum() < 15:
        return label, len(df), None, None, int(wet.sum()), "비 온 날 부족"

    X = model.design_matrix(df, weather=True)
    # 값이 하나뿐인 열은 상수라 const 와 겹친다. 빼야 수렴한다.
    varying = [c for c in X.columns if c == "const" or X[c].nunique() > 1]
    X = X[varying]
    fit = model.fit_tobit(X.values, df["log_crowd"].values, df["log_cap"].values)
    if not fit["converged"]:
        return label, len(df), None, None, int(wet.sum()), "수렴 실패"

    coef_any = float(fit["beta"][X.columns.get_loc("rain_any")]) if "rain_any" in X.columns else 0.0
    coef_amt = float(fit["beta"][X.columns.get_loc("rain_game")]) if "rain_game" in X.columns else 0.0
    return label, len(df), np.exp(coef_any) - 1, coef_amt, int(wet.sum()), ""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-season", default="2023")
    ap.add_argument("--keep-march", action="store_true",
                    help="3월도 포함한다. 그 구간 강수는 3시간 누적이라 기본은 제외")
    args = ap.parse_args()

    df = model.load_dataset(min_season=args.min_season)
    if df["rain_game"].notna().sum() == 0:
        raise SystemExit(
            "날씨 데이터가 없다.\n"
            "  1) https://apihub.kma.go.kr 에서 인증키를 받고 지상관측 ASOS 시간자료를 활용신청한다\n"
            "  2) KMA_API_KEY 를 환경변수로 넣고 python scripts/fetch_weather.py\n"
            "  3) python scripts/build_dataset.py"
        )

    if not args.keep_march:
        df = df[df["month"].astype(str) != "03"].copy()

    df["rain_game"] = pd.to_numeric(df["rain_game"], errors="coerce")
    df["rain_day"] = pd.to_numeric(df.get("rain_day"), errors="coerce")
    df["is_dome"] = pd.to_numeric(df["is_dome"], errors="coerce").fillna(0)
    weekend = df["dow"].isin(["토", "일"])

    groups = [
        ("전체", df),
        ("야외 구장", df[df["is_dome"] == 0]),
        ("돔 구장(고척)", df[df["is_dome"] == 1]),
        ("야외 주중", df[(df["is_dome"] == 0) & ~weekend]),
        ("야외 주말", df[(df["is_dome"] == 0) & weekend]),
    ]

    print("=== 비가 수요에 주는 영향 (요일, 팀, 구장, 시즌 통제) ===")
    print("  %-14s %6s %6s %9s %11s" % ("구간", "경기", "비온날", "비오면", "mm당"))
    for label, subset in groups:
        name, n, pct, per_mm, wet, why = rain_effect(subset, label)
        if pct is None:
            print("  %-14s %6d %6d %9s %11s" % (name, n, wet, why, "-"))
            continue
        print("  %-14s %6d %6d %8.1f%% %10.2f%%"
              % (name, n, wet, pct * 100, (np.exp(per_mm) - 1) * 100))

    print()
    print("  '비오면' 은 경기 시간대에 비가 온 날의 수요 변화, 'mm당' 은 강수량 1mm 가 더해질 때다.")

    # 경기 중에는 안 왔어도 그날 비가 왔으면 표를 덜 산다. 그 둘을 갈라 본다.
    wet_game = df["rain_game"].fillna(0) > 0
    wet_day = df["rain_day"].fillna(0) > 0
    print()
    print("=== 비가 언제 왔나 (%d경기) ===" % len(df))
    for label, mask in [
        ("경기 중에 왔다", wet_game),
        ("그날 왔지만 경기 중엔 안 왔다", wet_day & ~wet_game),
        ("하루 종일 안 왔다", ~wet_day),
    ]:
        subset = df[mask]
        if len(subset):
            print("  %-28s %5d경기  평균 점유율 %5.1f%%  매진율 %5.1f%%"
                  % (label, len(subset), subset["occupancy"].mean() * 100,
                     subset["sold_out"].mean() * 100))

    # 비가 어느 경로로 작동하는지 가르는 대조. 오가는 길의 날씨가 원인이라면
    # 지붕이 있어도 줄어야 한다. 돔에서 줄지 않는다면 남는 설명은 경기 자체가
    # 취소될 수 있다는 불확실성과 야외에서 비를 맞는 것이다.
    print()
    print("=== 지붕이 있을 때와 없을 때 ===")
    print("  %-12s %6s %8s %11s %11s %8s" % ("", "경기", "비온날", "비 온 날", "안 온 날", "차이"))
    for label, mask in [("야외 구장", df["is_dome"] == 0), ("돔 구장(고척)", df["is_dome"] == 1)]:
        subset = df[mask]
        w = subset[subset["rain_game"].fillna(0) > 0]
        d = subset[subset["rain_game"].fillna(0) <= 0]
        if len(w) < 5 or len(d) < 5:
            continue
        a, b = w["occupancy"].mean() * 100, d["occupancy"].mean() * 100
        print("  %-12s %6d %8d %10.1f%% %10.1f%% %+7.1f%%p"
              % (label, len(subset), len(w), a, b, a - b))
    print("  통제 없는 단순 비교다. 위 회귀와 달리 요일이나 팀 구성이 섞여 있다.")

    # 취소는 관중 데이터에 아예 없다. 비가 경기 자체를 날리는 쪽도 같이 본다.
    schedule = pd.read_csv(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "schedule.csv"))
    recent = schedule[schedule["season"].astype(str) >= str(args.min_season)]
    cancelled = recent["note"].astype(str).str.contains("우천").sum()
    print()
    print("  참고: %s 시즌 이후 편성 %d경기 중 우천취소 %d경기 (%.1f%%)"
          % (args.min_season, len(recent), cancelled, cancelled / len(recent) * 100))

    # 위까지는 열린 경기만 본 것이다. 비의 큰 몫이 거기에 안 잡힌다.
    sch = scheduled_with_rain(args.min_season, args.keep_march)
    print()
    cancellation_table(sch)
    print()
    selection_table(sch)


if __name__ == "__main__":
    main()
