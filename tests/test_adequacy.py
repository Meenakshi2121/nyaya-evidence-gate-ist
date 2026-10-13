import numpy as np
import pandas as pd

from nyaya_gate.adequacy import data_adequacy_check


def test_data_adequacy_requires_observed_dictionary_valid_values():
    data = pd.DataFrame(
        {
            "AGE": [70, 71, np.nan, 73],
            "SEX": ["M", "F", "M", "X"],
            "RDEF1": ["Y", "C", "N", "Y"],
        }
    )
    result = data_adequacy_check(data, ["AGE", "SEX", "RDEF1"])
    np.testing.assert_array_equal(result, [1, 1, 0, 0])


def test_data_adequacy_normalises_whitespace_and_case():
    data = pd.DataFrame({"SEX": [" m ", "f"], "AGE": [60, 61]})
    np.testing.assert_array_equal(data_adequacy_check(data, ["SEX", "AGE"]), [1, 1])
