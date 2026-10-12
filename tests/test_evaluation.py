import numpy as np

from nyaya_gate.evaluation import aipw_policy_risk, aipw_policy_scores, group_policy_gain


def test_aipw_risk_matches_mean_score():
    y = np.array([0, 1, 0, 1], dtype=float)
    a = np.array([0, 0, 1, 1])
    policy = np.ones(4, dtype=int)
    mu0 = np.full(4, 0.50)
    mu1 = np.full(4, 0.25)
    scores = aipw_policy_scores(y, a, policy, mu0, mu1, 0.5)
    assert np.isclose(aipw_policy_risk(y, a, policy, mu0, mu1, 0.5), scores.mean())


def test_group_gain_sign_is_intuitive():
    individualised = np.array([0.20, 0.30, 0.60])
    default = np.array([0.30, 0.40, 0.50])
    accepted = np.array([1, 1, 0])
    assert np.isclose(group_policy_gain(individualised, default, accepted), 0.10)

