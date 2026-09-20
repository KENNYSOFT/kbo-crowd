"""검열 회귀가 제대로 맞는지 확인하는 테스트.

python analysis/test_model.py 로 실행한다.

참값을 아는 합성 데이터를 쓰는 이유는, 실제 관중 데이터로는 '맞았는지'를
확인할 방법이 없기 때문이다. 매진 경기의 진짜 수요는 아무도 모른다.
그래서 답을 아는 문제를 만들어 그 답을 되찾는지 본다.
"""

import os
import sys

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model


def make_data(seed=7, n=400, censor_at=2.6):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    beta = np.array([2.0, 0.5, -0.3, 0.8])
    sigma = 0.6
    latent = X @ beta + rng.normal(scale=sigma, size=n)
    cap = np.full(n, censor_at)
    return X, np.minimum(latent, cap), cap, beta, sigma


def test_gradient():
    """해석적 기울기가 수치 미분과 맞는지. 틀린 기울기는 조용히 엉뚱한 답으로 간다."""
    X, y, cap, beta, sigma = make_data()
    k = X.shape[1]
    censored = y >= cap - 1e-9
    Xo, Xc, yo, uc = X[~censored], X[censored], y[~censored], cap[censored]

    def obj(p):
        b, ls = p[:k], float(np.clip(p[k], -8, 8))
        s = np.exp(ls)
        zo = (yo - Xo @ b) / s
        zc = (uc - Xc @ b) / s
        lsf = stats.norm.logsf(zc)
        total = (-ls - 0.5 * np.log(2 * np.pi) - 0.5 * zo ** 2).sum() + lsf.sum()
        r = np.exp(stats.norm.logpdf(zc) - lsf)
        grad = np.append(Xo.T @ (zo / s) + Xc.T @ (r / s), np.sum(zo ** 2 - 1) + np.sum(r * zc))
        return -total, -grad

    start = np.append(beta, np.log(sigma)) + 0.1
    err = optimize.check_grad(lambda p: obj(p)[0], lambda p: obj(p)[1], start)
    assert err < 1e-3, "기울기가 수치 미분과 어긋난다: %.3e" % err
    print("  기울기 오차 %.2e" % err)


def test_recovers_truth():
    """참 모수를 되찾는지, 그리고 OLS 보다 나은지."""
    X, y, cap, beta, sigma = make_data()
    fit = model.fit_tobit(X, y, cap)
    assert fit["converged"], "수렴하지 않았다: " + fit["message"]

    ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    tobit_err = np.abs(fit["beta"] - beta).mean()
    ols_err = np.abs(ols - beta).mean()

    assert tobit_err < ols_err, "Tobit 이 OLS 보다 나쁘다"
    assert tobit_err < 0.12, "참값에서 너무 멀다: %.3f" % tobit_err
    print("  평균 절대오차  Tobit %.3f < OLS %.3f" % (tobit_err, ols_err))

    # 검열을 무시하면 기울기가 0 쪽으로 눌린다. 그게 이 모델을 쓰는 이유다.
    assert np.abs(ols[1:]).sum() < np.abs(beta[1:]).sum(), "OLS 감쇠가 재현되지 않았다"


def test_sellout_probability():
    """매진 확률이 실제 검열 비율과 비슷하게 나오는지."""
    X, y, cap, _, _ = make_data()
    fit = model.fit_tobit(X, y, cap)
    p = model.sellout_probability(fit, X, cap)
    actual = (y >= cap - 1e-9).mean()
    assert abs(p.mean() - actual) < 0.05, "매진 확률 평균 %.3f vs 실제 %.3f" % (p.mean(), actual)
    print("  매진 확률 평균 %.3f (실제 %.3f)" % (p.mean(), actual))


def test_no_censoring_matches_ols():
    """검열이 없으면 결과가 최소제곱과 사실상 같아야 한다."""
    X, y, cap, _, _ = make_data(censor_at=99.0)
    fit = model.fit_tobit(X, y, cap)
    ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    gap = np.abs(fit["beta"] - ols).max()
    assert gap < 1e-3, "검열이 없는데 OLS 와 벌어진다: %.4f" % gap
    print("  무검열일 때 OLS 와 최대 차이 %.1e" % gap)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                print(name)
                fn()
            except AssertionError as exc:
                failures += 1
                print("  실패:", exc)
    print()
    print("모두 통과" if not failures else "%d건 실패" % failures)
    sys.exit(1 if failures else 0)
