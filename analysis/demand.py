"""검열 회귀로 팀별 티켓 파워와 잠재 수요를 추정한다.

python analysis/demand.py [--weather]

관중 수로 매긴 순위는 구장 크기에 오염돼 있다. 좌석이 작으면 수요가 아무리
커도 그 이상 팔 수 없기 때문이다. 여기서는 좌석 제약을 모델에 명시해 두고
'제약이 없었다면 얼마나 왔을까'를 추정해 순위를 다시 매긴다.

홈 효과와 원정 효과를 따로 둔다. 연고지에서 표를 파는 힘과 남의 구장에
팬을 데려가는 힘은 다른 것이라, 하나의 팀 계수로 합치면 둘 다 흐려진다.

시즌을 범주형으로 넣기 때문에 학습에 없던 시즌은 예측할 수 없다. KBO 관중은
2023년부터 해마다 크게 늘어서 시즌 효과를 빼면 모델이 망가지는데, 그 효과는
그 시즌 경기를 봐야 알 수 있다. 그래서 검증도 '다음 시즌'이 아니라 '같은
시즌의 뒷부분'으로 한다. 실제 쓰임새인 오늘 경기 예측이 딱 그 상황이다.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model


def team_effects(fit, X, prefix, all_teams):
    """더미 계수를 배수로 바꾼다.

    get_dummies(drop_first=True) 가 뺀 기준 팀은 계수가 0 이다. 그 팀이 특별해
    보이지 않도록 전체 평균을 빼서 '평균 팀 대비 배수'로 바꾼다.
    """
    prefix_ = prefix + "_"
    raw = {
        c[len(prefix_):]: float(fit["beta"][X.columns.get_loc(c)])
        for c in X.columns
        if c.startswith(prefix_)
    }
    for team in all_teams:  # 더미에 없는 팀이 기준 범주다
        raw.setdefault(team, 0.0)
    center = np.mean(list(raw.values()))
    return {team: float(np.exp(v - center)) for team, v in raw.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weather", action="store_true", help="날씨 변수를 함께 넣는다")
    ap.add_argument("--min-season", default="2023", help="이 시즌부터 학습 (기본 2023)")
    ap.add_argument("--holdout-from", help="이 날짜(YYYY-MM-DD) 이후 경기를 검증용으로 뺀다")
    args = ap.parse_args()

    df = model.load_dataset(min_season=args.min_season)
    if args.weather and df["temp"].notna().sum() == 0:
        print("날씨 데이터가 비어 있다. KMA_API_KEY 를 넣고 fetch_weather.py 를 먼저 돌릴 것.")
        args.weather = False

    train = df[df["date"] < args.holdout_from] if args.holdout_from else df
    X = model.design_matrix(train, weather=args.weather)
    fit = model.fit_tobit(X.values, train["log_crowd"].values, train["log_cap"].values)

    print("학습 %d경기 (검열 %d, %.1f%%) / 시즌 %s"
          % (fit["n"], fit["n_censored"], fit["n_censored"] / fit["n"] * 100,
             ", ".join(sorted(train["season"].unique()))))
    print("수렴 %s / sigma %.4f / 로그우도 %.1f"
          % ("성공" if fit["converged"] else "실패", fit["sigma"], fit["loglik"]))
    print()

    all_teams = sorted(set(train["home"]) | set(train["away"]))
    latent = np.exp(model.latent_demand(fit, X.values))

    for prefix, label in (("hometeam", "홈"), ("awayteam", "원정")):
        effects = team_effects(fit, X, prefix, all_teams)
        print("=== %s 티켓 파워 (전체 평균 = 1.00) ===" % label)
        for team, value in sorted(effects.items(), key=lambda kv: -kv[1]):
            bar = "#" * int(round(abs(value - 1) * 60))
            print("  %-4s %5.2f  %s" % (team, value, bar))
        print()

    # 관측 순위와 잠재 수요 순위를 나란히 둔다. 이 차이가 검열의 크기다.
    train = train.copy()
    train["latent"] = latent
    summary = (
        train.groupby("home")
        .agg(관측평균=("crowd", "mean"), 잠재수요=("latent", "mean"),
             매진율=("sold_out", "mean"), 좌석=("capacity", "median"))
        .sort_values("잠재수요", ascending=False)
    )
    summary["관측순위"] = summary["관측평균"].rank(ascending=False).astype(int)
    summary["잠재순위"] = summary["잠재수요"].rank(ascending=False).astype(int)
    summary["순위변화"] = summary["관측순위"] - summary["잠재순위"]

    print("=== 홈경기 수요: 관측 대비 추정 ===")
    print("  팀    좌석   관측평균(순위)   잠재수요(순위)  매진율  순위변화")
    for team, r in summary.iterrows():
        move = int(r["순위변화"])
        arrow = ("+%d" % move) if move > 0 else (str(move) if move < 0 else "-")
        print("  %-4s %6d  %7d(%2d)    %7d(%2d)   %5.1f%%   %s"
              % (team, r["좌석"], r["관측평균"], r["관측순위"],
                 r["잠재수요"], r["잠재순위"], r["매진율"] * 100, arrow))

    print()
    print("전체 평균: 관측 %d명 / 잠재 수요 %d명 (차이 %+.1f%%)"
          % (train["crowd"].mean(), train["latent"].mean(),
             (train["latent"].mean() / train["crowd"].mean() - 1) * 100))

    if args.holdout_from:
        test = df[df["date"] >= args.holdout_from]
        if len(test):
            Xt = model.design_matrix(test, weather=args.weather,
                                     reference=[c for c in X.columns if c != "const"])
            cap_t = test["log_cap"].values
            pred = np.exp(model.expected_observed(fit, Xt.values, cap_t))
            err = pred - test["crowd"].values
            prob = model.sellout_probability(fit, Xt.values, cap_t)
            baseline = train.groupby("home")["crowd"].mean()
            base_pred = test["home"].map(baseline).fillna(train["crowd"].mean()).values
            print()
            print("=== %s 이후 검증 (%d경기, 학습에서 제외) ===" % (args.holdout_from, len(test)))
            print("  평균절대오차 %d명 (홈팀 평균만 쓰는 기준선 %d명)"
                  % (np.abs(err).mean(), np.abs(base_pred - test["crowd"].values).mean()))
            print("  편향 %+d명" % err.mean())
            print("  매진 예측 평균 확률 %.3f vs 실제 매진율 %.3f"
                  % (prob.mean(), test["sold_out"].mean()))


if __name__ == "__main__":
    main()
