"""그 시점에 알 수 있던 것만으로 예측해 실제와 맞춰 본다.

python analysis/backtest.py [--start 2026-05-01]

모델이 얼마나 맞는지 말하려면 예측과 실제를 같은 자리에 놓아야 한다. 그런데
한 번 맞춘 모델로 과거를 되짚으면 그 경기의 결과가 이미 학습에 들어가 있어
실제보다 잘 맞는 것처럼 보인다. 그래서 주 단위로 앞으로 나아가며, 매번 그
주가 시작하기 전까지의 경기만으로 다시 학습한다. 실제 운영과 같은 순서다.

주 단위로 끊는 것은 비용 때문만은 아니다. 날마다 다시 맞추는 편이 이론상
낫지만 하루치 경기가 다섯 경기뿐이라 모수가 거의 움직이지 않고, 실제로도
모델을 매일 새로 맞춰 쓰지는 않는다.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model


def walk_forward(min_season="2023", start="2026-05-01", step_days=7,
                 keep_march=False, min_train=400):
    """start 부터 주 단위로 나아가며 그 주 경기를 예측하고 실제와 함께 돌려준다."""
    df = model.load_dataset(min_season=min_season)
    if not keep_march:
        df = df[df["month"].astype(str) != "03"].copy()
    later = df[df["date"] >= start]
    if later.empty:
        return pd.DataFrame()

    dates = sorted(later["date"].unique())
    rows, cursor = [], dates[0]
    while cursor <= dates[-1]:
        nxt = (pd.Timestamp(cursor) + pd.Timedelta(days=step_days)).strftime("%Y-%m-%d")
        train = df[df["date"] < cursor]
        test = df[(df["date"] >= cursor) & (df["date"] < nxt)]
        cursor = nxt
        if len(train) < min_train or test.empty:
            continue

        X = model.design_matrix(train, weather=False)
        fit = model.fit_tobit(X.values, train["log_crowd"].values, train["log_cap"].values)
        ref = [c for c in X.columns if c != "const"]
        Xt = model.design_matrix(test, weather=False, reference=ref)
        log_cap = np.log(test["capacity"].values)
        predicted = np.exp(model.expected_observed(fit, Xt.values, log_cap))
        prob = model.sellout_probability(fit, Xt.values, log_cap)

        for i, r in enumerate(test.itertuples()):
            rows.append({
                "date": r.date, "stadium": r.stadium, "home": r.home, "away": r.away,
                "seats": int(r.capacity), "actual": int(r.crowd),
                "predicted": float(predicted[i]), "prob": float(prob[i]),
                "soldOut": int(r.sold_out), "trainN": len(train),
            })
    return pd.DataFrame(rows)


def summary(out):
    """맞은 정도를 한 묶음으로 간추린다."""
    if out.empty:
        return None
    err = (out["predicted"] - out["actual"]).abs()
    return {
        "games": int(len(out)),
        "weeks": int(out["date"].str[:10].nunique()),
        "mae": float(err.mean()),
        "medae": float(err.median()),
        "maeSeat": float((err / out["seats"]).mean()),
        "selloutHit": float(((out["prob"] >= 0.5).astype(int) == out["soldOut"]).mean()),
        "brier": float(((out["prob"] - out["soldOut"]) ** 2).mean()),
        "over": float((out["predicted"] > out["actual"]).mean()),
        "from": out["date"].min(),
        "to": out["date"].max(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-season", default="2023")
    ap.add_argument("--start", default="2026-05-01", help="이 날짜부터 앞으로 나아간다")
    ap.add_argument("--step-days", type=int, default=7, help="며칠마다 다시 학습할지")
    ap.add_argument("--keep-march", action="store_true")
    args = ap.parse_args()

    out = walk_forward(args.min_season, args.start, args.step_days, args.keep_march)
    if out.empty:
        raise SystemExit("%s 이후 경기가 없다." % args.start)
    s = summary(out)

    print("=== 예측을 실제와 맞춰 본다 (%s ~ %s) ===" % (s["from"], s["to"]))
    print("  %d경기. 매주 그 주 이전 경기만으로 다시 학습했다." % s["games"])
    print()
    print("  평균 절대오차   %6.0f명" % s["mae"])
    print("  중앙값 오차     %6.0f명   (좌석 대비 평균 %.1f%%)" % (s["medae"], s["maeSeat"] * 100))
    print("  매진 맞힌 비율   %5.1f%%   (확률 50%% 를 경계로)" % (s["selloutHit"] * 100))
    print("  브라이어        %6.4f" % s["brier"])
    print("  예측이 실제보다 큰 경우 %.0f%%" % (s["over"] * 100))

    err = (out["predicted"] - out["actual"]).abs()
    print()
    print("  가장 많이 빗나간 경기")
    for r in out.reindex(err.sort_values(ascending=False).index).head(5).itertuples():
        print("    %s %-5s %-4s vs %-4s  예상 %6.0f  실제 %6d  (%+.0f)"
              % (r.date, r.stadium, r.away, r.home, r.predicted, r.actual,
                 r.predicted - r.actual))


if __name__ == "__main__":
    main()
