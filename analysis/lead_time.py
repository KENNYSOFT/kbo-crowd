"""티켓이 열리는 시점에 난이도를 예보하려면 무엇이 필요한가.

python analysis/lead_time.py

예매는 대체로 경기 일주일쯤 전에 열린다. 그때 난이도를 알려면 그 시점에
알 수 있는 것만으로 맞혀야 하는데, 그러면 두 가지가 걸린다. 날씨는 아직
모르고, 순위와 연승은 일주일 사이에 바뀐다. 둘 중 무엇이 실제로 비싼지
재는 것이 이 스크립트다.

결과를 먼저 적어 두면, 비싼 쪽은 날씨가 아니라 리드타임이다. 비는 관중을
분명히 깎지만 그 경로가 거의 당일 판매라서, 일주일 전에 팔리는 표는 날씨를
보고 팔리는 것이 아니다. 그래서 완벽한 예보를 가정해도 매진 예측은 나아지지
않는다. 근거는 아래 세 표에 있다.

rain_price.py 와 목적이 다르다. 그쪽은 비가 수요를 얼마나 깎는지(인과)를
재고, 여기서는 그 날씨를 알면 예측이 나아지는지를 잰다. 다른 질문이고
이 데이터에서는 답도 다르게 나온다.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model

RANK_FEATURES = ["rank", "wpct", "gb", "streak", "last10"]


def scores(p, y):
    """확률 예측을 세 가지로 잰다.

    브라이어와 로그손실은 낮을수록, AUC 는 높을수록 좋다. 셋을 함께 보는 것은
    AUC 가 순서만 보고 보정을 안 보기 때문이다. 확률값 자체가 틀려도 순서만
    맞으면 AUC 는 그대로라, 그것만 보면 나빠진 모델을 같다고 읽게 된다.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    npos, nneg = y.sum(), len(y) - y.sum()
    if not npos or not nneg:
        return brier, logloss, float("nan")
    auc = float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return brier, logloss, auc


def rank_history(df):
    """홈과 원정으로 흩어진 순위를 (팀, 날짜) 한 줄씩으로 편다."""
    parts = []
    for side in ["home", "away"]:
        cols = ["%s_%s" % (side, c) for c in RANK_FEATURES]
        part = df[["d", side] + cols].copy()
        part.columns = ["d", "team"] + RANK_FEATURES
        parts.append(part)
    return pd.concat(parts).sort_values("d").reset_index(drop=True)


def as_of(df, history, days):
    """각 경기의 순위를 days 일 전 최신값으로 되돌린다.

    티켓이 열리는 시점에 실제로 알 수 있는 것은 그 값이다. 경기 직전 순위로
    예보 성능을 재면 그때는 없던 정보를 쓴 셈이라 성능이 부풀려진다.
    """
    if days == 0:
        return df
    out = df.copy()
    for side in ["home", "away"]:
        asof = out[["d", side]].rename(columns={side: "team"}).copy()
        asof["cut"] = asof["d"] - pd.Timedelta(days=days)
        merged = pd.merge_asof(
            asof.sort_values("cut"), history,
            left_on="cut", right_on="d", by="team", direction="backward",
        ).sort_index()
        for col in RANK_FEATURES:
            out["%s_%s" % (side, col)] = merged[col].values
    return out


def load(min_season, keep_march=False):
    df = model.load_dataset(min_season=min_season)
    if not keep_march:
        # 11월부터 이듬해 3월까지 강수가 3시간 누적으로 와서 경기 시간대가 어긋난다.
        df = df[df["month"].astype(str) != "03"].copy()
    for col in ["rain_game", "rain_day", "is_dome"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["d"] = pd.to_datetime(df["date"])
    df["wet"] = df["rain_game"] > 0
    df["wet_day"] = df["rain_day"] > 0
    df["weekend"] = df["dow"].isin(["토", "일"])
    return df.reset_index(drop=True)


def when_does_rain_work(df):
    """비가 어느 단계에서 작동하는지 본다.

    예매가 일찍 끝나는 주말과 당일 판매 비중이 큰 주중을 견주면, 비가 표를
    미리 사는 단계에서 작동하는지 당일 단계에서 작동하는지 갈린다.
    """
    outdoor = df[df["is_dome"] == 0].copy()
    X = model.design_matrix(outdoor, weather=False)
    fit = model.fit_tobit(X.values, outdoor["log_crowd"].values, outdoor["log_cap"].values)
    outdoor["tight"] = model.latent_demand(fit, X.values) - outdoor["log_cap"]

    print("=== 비가 무엇을 바꾸나 (야외 %d경기, 기저 수요 4분위) ===" % len(outdoor))
    print("  기저 수요는 날씨를 뺀 모델이 본 것이라, 경기를 비와 무관하게 가른다.")
    print("  %-10s %5s %5s %11s %11s" % ("수요 수준", "경기", "비온날", "점유율 차이", "매진율 차이"))
    outdoor["band"] = pd.qcut(outdoor["tight"], 4, labels=["여유", "보통", "빠듯", "매우 빠듯"])
    for band, g in outdoor.groupby("band", observed=True):
        wet, dry = g[g["wet"]], g[~g["wet"]]
        if len(wet) < 5:
            continue
        print("  %-10s %5d %5d %10.1f%%p %10.1f%%p"
              % (band, len(g), len(wet),
                 (wet["occupancy"].mean() - dry["occupancy"].mean()) * 100,
                 (wet["sold_out"].mean() - dry["sold_out"].mean()) * 100))

    hot = outdoor[outdoor["tight"] > outdoor["tight"].median()]
    print()
    print("=== 그 비는 언제 작동했나 (수요가 빠듯한 절반 %d경기) ===" % len(hot))
    print("  %-6s %5s %5s %9s %9s %9s" % ("", "경기", "비온날", "매진 비", "매진 맑음", "차이"))
    for is_weekend, label in [(False, "주중"), (True, "주말")]:
        g = hot[hot["weekend"] == is_weekend]
        wet, dry = g[g["wet"]], g[~g["wet"]]
        if len(wet) < 5:
            continue
        print("  %-6s %5d %5d %8.1f%% %8.1f%% %+8.1f%%p"
              % (label, len(g), len(wet), wet["sold_out"].mean() * 100,
                 dry["sold_out"].mean() * 100,
                 (wet["sold_out"].mean() - dry["sold_out"].mean()) * 100))
    print("  주말은 표가 일찍 팔린다. 거기서 비 효과가 작으면 비는 당일에 작동한 것이다.")

    dome = df[df["is_dome"] == 1]
    wet, dry = dome[dome["wet"]], dome[~dome["wet"]]
    if len(wet) >= 5:
        print()
        print("  대조: 돔 %d경기(비 %d경기) 매진율 %.1f%% vs %.1f%%. 지붕이 있으면 비가 깎지 않는다."
              % (len(dome), len(wet), wet["sold_out"].mean() * 100, dry["sold_out"].mean() * 100))


def does_weather_help(df, cut):
    """날씨를 넣으면 매진 예측이 나아지는지 본다.

    검증에 쓰는 날씨는 실제 관측값이다. 즉 예보가 100% 맞았을 때의 상한을
    재는 것이고, 실제 예보로는 이보다 나을 수 없다.
    """
    train, test = df[df["date"] < cut], df[df["date"] >= cut]
    y = test["sold_out"].values.astype(float)
    print("=== 날씨를 알면 매진 예측이 나아지나 (%s 이후 %d경기) ===" % (cut, len(test)))
    print("  %-22s %9s %9s %7s" % ("", "브라이어", "로그손실", "AUC"))
    for label, weather in [("날씨 없음", False), ("날씨 있음(완벽 예보)", True)]:
        X = model.design_matrix(train, weather=weather)
        fit = model.fit_tobit(X.values, train["log_crowd"].values, train["log_cap"].values)
        ref = [c for c in X.columns if c != "const"]
        Xt = model.design_matrix(test, weather=weather, reference=ref)
        p = model.sellout_probability(fit, Xt.values, np.log(test["capacity"].values))
        print("  %-22s %9.4f %9.4f %7.3f" % ((label,) + scores(p, y)))
    brier, logloss, _ = scores(np.full(len(y), train["sold_out"].mean()), y)
    print("  %-22s %9.4f %9.4f %7s" % ("(기준선: 평균으로 찍기)", brier, logloss, "-"))
    print("  완벽한 예보로도 나아지지 않으면, 부정확한 실제 예보는 볼 것도 없다.")


def how_early(df, cut, days_list):
    """며칠 전에 예보하느냐가 성능을 얼마나 깎는지 본다."""
    train = df[df["date"] < cut]
    X = model.design_matrix(train, weather=False)
    fit = model.fit_tobit(X.values, train["log_crowd"].values, train["log_cap"].values)
    ref = [c for c in X.columns if c != "const"]
    history = rank_history(df)

    print("=== 며칠 전에 예보하나 (%s 이후 검증, 날씨 없음) ===" % cut)
    print("  %-10s %9s %9s %7s" % ("예보 시점", "브라이어", "로그손실", "AUC"))
    for days in days_list:
        test = as_of(df, history, days)
        test = test[test["date"] >= cut]
        Xt = model.design_matrix(test, weather=False, reference=ref)
        p = model.sellout_probability(fit, Xt.values, np.log(test["capacity"].values))
        label = "경기 당일" if days == 0 else "%d일 전" % days
        print("  %-10s %9.4f %9.4f %7.3f"
              % ((label,) + scores(p, test["sold_out"].values.astype(float))))

    week = as_of(df, history, 7)
    print("  7일 전 값은 당일과 순위가 평균 %.1f계단, 연승이 평균 %.1f경기 차이 난다."
          % ((week["home_rank"] - df["home_rank"]).abs().mean(),
             (week["home_streak"] - df["home_streak"]).abs().mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-season", default="2023")
    ap.add_argument("--holdout-from", default="2026-07-01", help="이 날짜부터를 검증에 쓴다")
    ap.add_argument("--keep-march", action="store_true")
    args = ap.parse_args()

    df = load(args.min_season, args.keep_march)
    if df["rain_game"].notna().sum() == 0:
        raise SystemExit("날씨 데이터가 없다. scripts/fetch_weather.py 를 먼저 돌린다.")
    if len(df[df["date"] >= args.holdout_from]) < 100:
        raise SystemExit("%s 이후 경기가 100개도 안 된다. --holdout-from 을 앞당긴다." % args.holdout_from)

    when_does_rain_work(df)
    print()
    does_weather_help(df, args.holdout_from)
    print()
    how_early(df, args.holdout_from, [0, 3, 7, 14])


if __name__ == "__main__":
    main()
