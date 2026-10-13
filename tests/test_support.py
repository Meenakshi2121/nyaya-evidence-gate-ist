import numpy as np
import pandas as pd

from nyaya_gate.support import GowerReferenceSupport, _GowerEncoder


def test_gower_encoding_gives_each_original_variable_equal_weight():
    training = pd.DataFrame(
        {
            "age": [0.0, 10.0],
            "sex": ["F", "M"],
        }
    )
    encoder = _GowerEncoder(("age",), ("sex",)).fit(training)
    transformed = encoder.transform(training)
    distance = np.abs(transformed[0] - transformed[1]).sum()
    assert np.isclose(distance, 1.0)


def test_reference_support_flags_a_distant_case():
    arm = np.tile([0, 1], 6)
    training = pd.DataFrame(
        {
            "age": np.tile([50.0, 51.0, 52.0], 4),
            "sex": np.tile(["F", "M"], 6),
        }
    )
    support = GowerReferenceSupport(
        numeric_columns=("age",),
        categorical_columns=("sex",),
        n_neighbors=2,
        quantile=0.8,
        n_jobs=1,
    ).fit(training, arm)
    held_out = pd.DataFrame({"age": [51.0, 100.0], "sex": ["F", "F"]})
    scores = support.score(held_out)
    assert scores[0] < scores[1]
    assert support.check(held_out)[1] == 0
