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
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model


def rain_effect(df, label):
    """그 부분집합에서 비 계수를 배수로 낸다."""
    if len(df) < 150 or df["rain_game"].notna().sum() < 30:
        return label, len(df), None, None, None
    wet = (pd.to_numeric(df["rain_game"], errors="coerce").fillna(0) > 0)
    if wet.sum() < 15 or (~wet).sum() < 15:
        return label, len(df), None, None, None

    X = model.design_matrix(df, weather=True)
    fit = model.fit_tobit(X.values, df["log_crowd"].values, df["log_cap"].values)
    if not fit["converged"]:
        return label, len(df), None, None, None

    coef_any = float(fit["beta"][X.columns.get_loc("rain_any")]) if "rain_any" in X.columns else 0.0
    coef_amt = float(fit["beta"][X.columns.get_loc("rain_game")]) if "rain_game" in X.columns else 0.0
    return label, len(df), np.exp(coef_any) - 1, coef_amt, int(wet.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-season", default="2023")
    args = ap.parse_args()

    df = model.load_dataset(min_season=args.min_season)
    if df["rain_game"].notna().sum() == 0:
        raise SystemExit(
            "날씨 데이터가 없다.\n"
            "  1) https://apihub.kma.go.kr 에서 인증키를 받는다\n"
            "  2) KMA_API_KEY 를 환경변수로 넣고 python scripts/fetch_weather.py\n"
            "  3) python scripts/build_dataset.py"
        )

    df["rain_game"] = pd.to_numeric(df["rain_game"], errors="coerce")
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
        name, n, pct, per_mm, wet = rain_effect(subset, label)
        if pct is None:
            print("  %-14s %6d %6s %9s %11s" % (name, n, "-", "표본 부족", "-"))
            continue
        print("  %-14s %6d %6d %8.1f%% %10.2f%%"
              % (name, n, wet, pct * 100, (np.exp(per_mm) - 1) * 100))

    print()
    print("  '비오면' 은 경기 시간대에 비가 온 날의 수요 변화, 'mm당' 은 강수량 1mm 가 더해질 때다.")

    # 취소는 관중 데이터에 아예 없다. 비가 경기 자체를 날리는 쪽도 같이 본다.
    schedule = pd.read_csv(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "schedule.csv"))
    recent = schedule[schedule["season"].astype(str) >= str(args.min_season)]
    cancelled = recent["note"].astype(str).str.contains("우천").sum()
    print()
    print("  참고: %s 시즌 이후 편성 %d경기 중 우천취소 %d경기 (%.1f%%)"
          % (args.min_season, len(recent), cancelled, cancelled / len(recent) * 100))


if __name__ == "__main__":
    main()
