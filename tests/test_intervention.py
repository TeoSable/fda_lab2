import numpy as np
import pytest

from emlab.coupling import transient
from emlab.intervention import (
    Regression,
    conditional_noise,
    marginal_w1,
    residual_bootstrap,
    rlc_current,
)


@pytest.mark.parametrize("theta", [[12.0, 4.0], [30.0, 8.0], [0.0, 4.0]])
def test_exact_solution_agrees_with_original_ode(theta):
    t = np.linspace(0, 2e-6, 151)
    expected = transient(t, theta[0] * 1e-9, theta[1], rtol=1e-10)[3]
    np.testing.assert_allclose(
        rlc_current(t, theta), expected, atol=2e-10, rtol=2e-7
    )


def test_wasserstein_known_translation_and_identity():
    a = np.arange(100.0).reshape(50, 2)
    np.testing.assert_allclose(marginal_w1(a, a)[0], 0)
    np.testing.assert_allclose(marginal_w1(a, a + [3.0, -4.0])[0], [3.0, 4.0])


def test_conditional_variance_and_zero_control():
    rng = np.random.default_rng(5)
    scale = np.repeat([1.0, 3.0], 100000)
    y = rng.normal(size=len(scale)) * scale
    z = rng.normal(size=len(scale))
    np.testing.assert_array_equal(conditional_noise(y, scale, 0, z), y)
    changed = conditional_noise(y, scale, 2, z)
    for sl, s in [(slice(0, 100000), 1.0), (slice(100000, None), 3.0)]:
        assert abs(np.var(changed[sl]) / (5 * s * s) - 1) < 0.02


def test_bootstrap_refits_coefficients_with_fixed_design():
    rng = np.random.default_rng(40)
    x = np.linspace(-1, 1, 100)
    scale = 0.05 * (2 + x)
    y = 2 + 3 * x + rng.normal(size=len(x)) * scale
    model = Regression(
        x,
        y,
        scale,
        ("intercept", "slope"),
        lambda t, p: p[0] + p[1] * t,
        np.array([1.0, 1.0]),
        ([-10.0, -10.0], [10.0, 10.0]),
    )
    indices = rng.integers(0, len(x), (60, len(x)))
    draws, point, diag = residual_bootstrap(model, y, indices)
    assert draws.shape == (60, 2)
    assert np.all(abs(point - [2, 3]) < 0.05)
    assert np.all(draws.std(axis=0) > 0.001)
    # Reusing the resampling plan must reproduce the distribution exactly.
    repeated, _, _ = residual_bootstrap(model, y, indices)
    np.testing.assert_array_equal(draws, repeated)
    assert diag["boundary_fraction"] == 0
