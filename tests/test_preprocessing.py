import numpy as np
import pandas as pd

from nyaya_gate.preprocessing import FoldPreprocessor


def test_fold_preprocessor_imputes_and_handles_unseen_categories():
    training = pd.DataFrame(
        {"age": [60.0, np.nan, 80.0], "sex": ["F", "M", None]}
    )
    held_out = pd.DataFrame({"age": [np.nan], "sex": ["X"]})
    preprocessor = FoldPreprocessor(("age",), ("sex",)).fit(training)
    transformed = preprocessor.transform(held_out)
    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()
    names = preprocessor.get_feature_names_out().tolist()
    assert "age__missing" in names
    assert "sex__missing" in names
