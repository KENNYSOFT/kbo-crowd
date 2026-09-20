"""관중 수요 모델. 검열 회귀(Tobit)와 그 입력을 만드는 코드.

왜 보통 회귀를 쓰지 않는가. 매진한 경기의 관중 수는 수요가 아니라 좌석 수다.
표가 다 팔린 날 몇 명이 더 오고 싶었는지는 기록에 남지 않는다. 전체의 3할쯤이
그렇게 상한에 눌려 있어서, 최소제곱법으로 맞추면 인기 있는 팀일수록 수요를
체계적으로 낮게 본다. 한화가 대표적이다. 평균 관중은 중하위권인데 매진율은
1위인 것은 대전 구장이 작아서이지 수요가 작아서가 아니다.

Tobit 은 관측값을 두 종류로 나눠 우도를 쓴다. 매진이 아닌 경기는 값 자체가
수요이므로 정규분포 밀도를 쓰고, 매진한 경기는 '수요가 최소한 좌석 수 이상'
이라는 사실만 알려주므로 그 위쪽 꼬리 확률을 쓴다. 그래서 잘린 부분을 메꾼
수요를 추정할 수 있다.

교과서의 Tobit 은 모든 관측이 같은 지점에서 잘린다고 보지만 여기서는 구장마다,
시즌마다 상한이 다르다. 그래서 관측별 검열점을 받도록 썼다.
"""

import sys

import numpy as np
from scipy import optimize, stats

# Windows 콘솔 기본 코드페이지로는 한국어 출력이 깨진다.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# 모델에 넣을 연속형 피처. 없는 값은 중앙값으로 채운다.
NUMERIC = [
    "home_rank", "away_rank", "home_wpct", "away_wpct",
    "home_streak", "away_streak", "home_last10", "away_last10",
    "gb_from_playoff_line", "season_progress",
]
# 범주형 컬럼과 더미 접두사. 접두사를 컬럼명 그대로 두면 home_rank 같은 연속형
# 변수가 home_ 로 시작해서 팀 더미와 섞인다.
CATEGORICAL = [("home", "hometeam"), ("away", "awayteam"), ("dow", "dow"),
               ("month", "month"), ("season", "season")]


def design_matrix(df, weather=False, reference=None):
    """설계행렬을 만든다.

    범주형은 더미로 펴되 기준 수준 하나를 뺀다(다중공선성 회피).
    reference 를 주면 학습 때 쓴 열 구성을 그대로 재현한다. 예측 시점에
    없는 범주가 있어도 열이 어긋나지 않게 하려는 것.
    """
    import pandas as pd

    cols = {}
    for name in NUMERIC:
        if name in df.columns:
            v = pd.to_numeric(df[name], errors="coerce")
            cols[name] = v.fillna(v.median() if v.notna().any() else 0.0)

    if weather:
        for name in ["temp", "rain_game", "humid", "wind", "cloud"]:
            if name in df.columns:
                v = pd.to_numeric(df[name], errors="coerce")
                cols[name] = v.fillna(v.median() if v.notna().any() else 0.0)
        if "rain_game" in cols:
            # 비는 양보다 '왔는가'가 더 세게 작동할 수 있어 둘 다 넣는다.
            cols["rain_any"] = (cols["rain_game"] > 0).astype(float)

    X = pd.DataFrame(cols, index=df.index)

    # 기준 범주를 떨어뜨리는 것은 학습 때 한 번뿐이다. 예측할 때도 drop_first 를
    # 걸면 그 데이터에 있는 범주를 기준으로 다시 떨어뜨리게 되는데, 예측 대상이
    # 한 시즌뿐이면 시즌 더미가 통째로 사라진 뒤 reindex 로 0 이 채워져 기준
    # 시즌으로 예측해 버린다. 조용히 틀리는 종류라 열 구성은 reference 가 정한다.
    predicting = reference is not None
    for name, prefix in CATEGORICAL:
        if name not in df.columns:
            continue
        dummies = pd.get_dummies(
            df[name].astype(str), prefix=prefix, drop_first=not predicting
        )
        X = pd.concat([X, dummies], axis=1)

    X = X.astype(float)
    if predicting:
        X = X.reindex(columns=reference, fill_value=0.0)
    X.insert(0, "const", 1.0)
    return X


def fit_tobit(X, y, upper, maxiter=2000):
    """상한에서 오른쪽으로 잘린 회귀를 최대우도로 맞춘다.

    y 는 관측값, upper 는 관측별 검열점, 둘 다 로그 스케일을 가정한다.
    y >= upper 인 관측을 검열로 본다.

    기울기를 해석적으로 준다. 수치 미분에 맡기면 더미 변수가 수십 개인 설계에서
    반복 한도 안에 수렴하지 못한다. 검열 쪽 항에 나오는 pdf/sf 는 꼬리로 갈수록
    0/0 이 되므로 로그 차이의 지수로 계산해 안정성을 지킨다.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    upper = np.asarray(upper, dtype=float)
    censored = y >= upper - 1e-9
    observed = ~censored
    n, k = X.shape

    Xo, Xc = X[observed], X[censored]
    yo, uc = y[observed], upper[censored]

    def objective(params):
        beta, log_sigma = params[:k], float(np.clip(params[k], -8, 8))
        sigma = np.exp(log_sigma)

        mu_o = Xo @ beta
        z_o = (yo - mu_o) / sigma
        ll_o = -log_sigma - 0.5 * np.log(2 * np.pi) - 0.5 * z_o ** 2

        mu_c = Xc @ beta
        z_c = (uc - mu_c) / sigma
        log_sf = stats.norm.logsf(z_c)
        ll_c = log_sf

        total = ll_o.sum() + ll_c.sum()
        if not np.isfinite(total):
            return 1e12, np.zeros(k + 1)

        # 역밀스비. 꼬리에서도 안전하게 exp(logpdf - logsf) 로 낸다.
        ratio = np.exp(stats.norm.logpdf(z_c) - log_sf)

        grad_beta = Xo.T @ (z_o / sigma) + Xc.T @ (ratio / sigma)
        grad_logsigma = np.sum(z_o ** 2 - 1.0) + np.sum(ratio * z_c)

        grad = np.append(grad_beta, grad_logsigma)
        return -total, -grad

    # 최소제곱 추정을 출발점으로 삼으면 수렴이 빠르다.
    beta0, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta0
    start = np.append(beta0, np.log(max(resid.std(), 1e-3)))

    res = optimize.minimize(
        objective, start, jac=True, method="L-BFGS-B", options={"maxiter": maxiter}
    )
    return {
        "beta": res.x[:k],
        "sigma": float(np.exp(res.x[k])),
        "loglik": float(-res.fun),
        "converged": bool(res.success),
        "message": str(res.message),
        "n_censored": int(censored.sum()),
        "n": n,
    }


def latent_demand(fit, X):
    """모델이 보는 잠재 수요(로그 스케일). 좌석 제약이 없었다면의 기댓값."""
    return np.asarray(X, dtype=float) @ fit["beta"]


def expected_observed(fit, X, upper):
    """실제로 관측될 값의 기댓값. 상한에서 잘리는 것을 반영한다."""
    mu = latent_demand(fit, X)
    sigma = fit["sigma"]
    z = (upper - mu) / sigma
    # E[min(Y, c)] = mu + sigma*(-pdf) ... 잘린 정규분포의 기댓값
    cdf = stats.norm.cdf(z)
    pdf = stats.norm.pdf(z)
    return mu * cdf - sigma * pdf + upper * (1 - cdf)


def sellout_probability(fit, X, upper):
    """그 경기가 매진될 확률."""
    mu = latent_demand(fit, X)
    return stats.norm.sf((upper - mu) / fit["sigma"])


def load_dataset(path=None, min_season=None, drop_restricted=True):
    """dataset.csv 를 읽어 모델에 쓸 형태로 돌려준다.

    2021년과 2022년 상당 기간은 코로나 입장 제한이 걸려 있어서, 그때의 상한은
    구장 크기가 아니라 방역 지침이다. 수요를 재는 데 쓰면 안 되므로 기본으로 뺀다.
    판정은 시즌 평균 점유율이 아니라 그 시즌 그 구장의 상한이 나중 시즌 대비
    얼마나 눌렸는지로 한다.
    """
    import os
    import pandas as pd

    if path is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "data", "dataset.csv")
    df = pd.read_csv(path, dtype={"season": str, "month": str})
    df = df[df["capacity"].notna() & (df["capacity"] > 0)].copy()

    if drop_restricted:
        peak = df.groupby("stadium")["capacity"].transform("max")
        df = df[df["capacity"] >= peak * 0.9].copy()
    if min_season:
        df = df[df["season"] >= str(min_season)].copy()

    df["log_crowd"] = np.log(df["crowd"].clip(lower=1))
    df["log_cap"] = np.log(df["capacity"])
    return df.reset_index(drop=True)
