import numpy as np
import pytest

from jax_dsmc.gradient_validation import paired_report


def test_equivalence_does_not_accept_unresolved_noise():
    x = np.tile([-10.0, 10.0], 16)
    assert paired_report(x, -x)["status"] == "inconclusive"
    assert paired_report(x, x)["status"] == "equivalent"
    assert paired_report(x + 1, x)["status"] == "different"


def test_invalid_seed_is_not_silently_discarded():
    with pytest.raises(ValueError, match="cannot be dropped"):
        paired_report([1.0, np.nan], [1.0, 1.0])
