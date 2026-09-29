"""
tests/test_analytics.py
-----------------------
Unit tests for the analytics module.

Run with:
    pytest tests/ -v

Or with coverage:
    pytest tests/ -v --tb=short
"""

import pytest
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analytics import (
    validate_data,
    representation_by_level,
    pay_gap_overall,
    pay_gap_by_level,
    pay_gap_by_department,
    promotion_rates,
    attrition_rates,
    train_promotion_model,
    fairness_metrics,
    REQUIRED_COLUMNS,
    LEVEL_ORDER,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Minimal but valid employee DataFrame for testing."""
    np.random.seed(42)
    n = 300
    genders = np.where(np.random.rand(n) > 0.5, "Male", "Female")
    levels = np.random.choice(LEVEL_ORDER, n)
    depts = np.random.choice(["Engineering", "HR", "Sales", "Finance"], n)
    edus = np.random.choice(["Bachelor", "Master", "PhD"], n)
    return pd.DataFrame({
        "gender":               genders,
        "annual_salary":        np.random.randint(400_000, 2_000_000, n),
        "job_level":            levels,
        "education":            edus,
        "department":           depts,
        "age":                  np.random.randint(22, 60, n),
        "experience_years":     np.random.randint(1, 30, n),
        "performance_rating":   np.round(np.random.uniform(1.0, 5.0, n), 1),
        "promoted_last_cycle":  np.random.randint(0, 2, n),
        "exited_last_12mo":     np.random.randint(0, 2, n),
    })


# ---------------------------------------------------------------------------
# validate_data
# ---------------------------------------------------------------------------

class TestValidateData:

    def test_valid_data_returns_true(self, sample_df):
        assert validate_data(sample_df) is True

    def test_missing_column_raises(self, sample_df):
        df = sample_df.drop(columns=["gender"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_data(df)

    def test_invalid_gender_value_raises(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "gender"] = "NonBinary"
        with pytest.raises(ValueError, match="Unexpected gender values"):
            validate_data(df)

    def test_null_in_salary_raises(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "annual_salary"] = np.nan
        with pytest.raises(ValueError, match="Null values"):
            validate_data(df)

    def test_zero_salary_raises(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "annual_salary"] = 0
        with pytest.raises(ValueError, match="non-positive"):
            validate_data(df)

    def test_negative_salary_raises(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "annual_salary"] = -5000
        with pytest.raises(ValueError, match="non-positive"):
            validate_data(df)

    def test_non_binary_promoted_raises(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "promoted_last_cycle"] = 99
        with pytest.raises(ValueError, match="binary"):
            validate_data(df)

    def test_non_binary_exited_raises(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "exited_last_12mo"] = 2
        with pytest.raises(ValueError, match="binary"):
            validate_data(df)

    def test_multiple_errors_reported_together(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "gender"] = "X"
        df.loc[1, "annual_salary"] = -100
        with pytest.raises(ValueError) as exc_info:
            validate_data(df)
        msg = str(exc_info.value)
        assert "gender" in msg.lower() or "salary" in msg.lower()


# ---------------------------------------------------------------------------
# representation_by_level
# ---------------------------------------------------------------------------

class TestRepresentationByLevel:

    def test_returns_tuple_of_two_dataframes(self, sample_df):
        result = representation_by_level(sample_df)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], pd.DataFrame)

    def test_index_matches_level_order(self, sample_df):
        counts, pct = representation_by_level(sample_df)
        assert list(counts.index) == LEVEL_ORDER

    def test_percentages_sum_to_100(self, sample_df):
        _, pct = representation_by_level(sample_df)
        row_sums = pct.sum(axis=1).dropna()
        np.testing.assert_allclose(row_sums.values, 100.0, atol=1e-6)

    def test_columns_are_gender_labels(self, sample_df):
        counts, _ = representation_by_level(sample_df)
        assert set(counts.columns).issubset({"Male", "Female"})


# ---------------------------------------------------------------------------
# pay_gap_overall
# ---------------------------------------------------------------------------

class TestPayGapOverall:

    def test_returns_dict_with_all_keys(self, sample_df):
        result = pay_gap_overall(sample_df)
        expected = {
            "male_avg_salary", "female_avg_salary", "gap_pct",
            "t_stat", "p_value", "significant_at_5pct",
        }
        assert expected.issubset(result.keys())

    def test_salaries_are_positive(self, sample_df):
        result = pay_gap_overall(sample_df)
        assert result["male_avg_salary"] > 0
        assert result["female_avg_salary"] > 0

    def test_gap_pct_is_float(self, sample_df):
        result = pay_gap_overall(sample_df)
        assert isinstance(result["gap_pct"], float)

    def test_significant_at_5pct_is_bool(self, sample_df):
        result = pay_gap_overall(sample_df)
        assert isinstance(result["significant_at_5pct"], bool)

    def test_p_value_between_0_and_1(self, sample_df):
        result = pay_gap_overall(sample_df)
        assert 0.0 <= result["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# pay_gap_by_level
# ---------------------------------------------------------------------------

class TestPayGapByLevel:

    def test_returns_dataframe(self, sample_df):
        result = pay_gap_by_level(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_gap_pct_column(self, sample_df):
        result = pay_gap_by_level(sample_df)
        assert "gap_pct" in result.columns

    def test_index_is_level_order(self, sample_df):
        result = pay_gap_by_level(sample_df)
        assert list(result.index) == LEVEL_ORDER


# ---------------------------------------------------------------------------
# pay_gap_by_department
# ---------------------------------------------------------------------------

class TestPayGapByDepartment:

    def test_returns_dataframe(self, sample_df):
        result = pay_gap_by_department(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_gap_columns(self, sample_df):
        result = pay_gap_by_department(sample_df)
        assert "gap_pct" in result.columns
        assert "gap_abs" in result.columns

    def test_index_is_department_names(self, sample_df):
        result = pay_gap_by_department(sample_df)
        assert result.index.name == "department"


# ---------------------------------------------------------------------------
# promotion_rates
# ---------------------------------------------------------------------------

class TestPromotionRates:

    def test_returns_series_and_dict(self, sample_df):
        rates, test_stats = promotion_rates(sample_df)
        assert isinstance(rates, pd.Series)
        assert isinstance(test_stats, dict)

    def test_chi_square_keys_present(self, sample_df):
        _, test_stats = promotion_rates(sample_df)
        assert "chi2" in test_stats
        assert "p_value" in test_stats

    def test_rates_between_0_and_100(self, sample_df):
        rates, _ = promotion_rates(sample_df)
        assert (rates >= 0).all()
        assert (rates <= 100).all()

    def test_chi2_is_non_negative(self, sample_df):
        _, test_stats = promotion_rates(sample_df)
        assert test_stats["chi2"] >= 0


# ---------------------------------------------------------------------------
# attrition_rates
# ---------------------------------------------------------------------------

class TestAttritionRates:

    def test_returns_series(self, sample_df):
        result = attrition_rates(sample_df)
        assert isinstance(result, pd.Series)

    def test_values_in_valid_range(self, sample_df):
        result = attrition_rates(sample_df)
        assert (result >= 0).all()
        assert (result <= 100).all()


# ---------------------------------------------------------------------------
# train_promotion_model
# ---------------------------------------------------------------------------

class TestTrainPromotionModel:

    @pytest.fixture(scope="class")
    def model_output(self, sample_df):
        return train_promotion_model(sample_df)

    def test_returns_three_elements(self, sample_df):
        out = train_promotion_model(sample_df)
        assert len(out) == 3

    def test_result_df_has_required_columns(self, sample_df):
        _, result_df, _ = train_promotion_model(sample_df)
        required = {"gender", "actual", "predicted", "pred_prob"}
        assert required.issubset(result_df.columns)

    def test_model_results_contains_both_models(self, sample_df):
        _, _, model_results = train_promotion_model(sample_df)
        assert "Logistic Regression" in model_results
        assert "Random Forest" in model_results

    def test_accuracy_in_valid_range(self, sample_df):
        _, _, model_results = train_promotion_model(sample_df)
        for name, res in model_results.items():
            assert 0.0 <= res["accuracy"] <= 1.0, f"{name}: accuracy out of range"

    def test_auc_in_valid_range(self, sample_df):
        _, _, model_results = train_promotion_model(sample_df)
        for name, res in model_results.items():
            assert 0.0 <= res["auc"] <= 1.0, f"{name}: AUC out of range"

    def test_predictions_are_binary(self, sample_df):
        _, result_df, _ = train_promotion_model(sample_df)
        assert set(result_df["predicted"].unique()).issubset({0, 1})

    def test_pred_prob_between_0_and_1(self, sample_df):
        _, result_df, _ = train_promotion_model(sample_df)
        assert (result_df["pred_prob"] >= 0).all()
        assert (result_df["pred_prob"] <= 1).all()


# ---------------------------------------------------------------------------
# fairness_metrics
# ---------------------------------------------------------------------------

class TestFairnessMetrics:

    @pytest.fixture
    def fm_result(self, sample_df):
        _, result_df, _ = train_promotion_model(sample_df)
        return fairness_metrics(result_df)

    def test_returns_expected_top_level_keys(self, fm_result):
        required = {
            "predicted_promotion_rate_male",
            "predicted_promotion_rate_female",
            "disparate_impact_ratio",
            "passes_four_fifths_rule",
            "statistical_parity_difference",
            "equalized_odds",
        }
        assert required.issubset(fm_result.keys())

    def test_equalized_odds_has_expected_keys(self, fm_result):
        eq = fm_result["equalized_odds"]
        required = {
            "tpr_male", "tpr_female", "tpr_gap",
            "fpr_male", "fpr_female", "fpr_gap",
            "passes_equalized_odds_01",
        }
        assert required.issubset(eq.keys())

    def test_dir_is_positive(self, fm_result):
        assert fm_result["disparate_impact_ratio"] > 0

    def test_passes_four_fifths_is_bool(self, fm_result):
        assert isinstance(fm_result["passes_four_fifths_rule"], bool)

    def test_passes_equalized_odds_is_bool(self, fm_result):
        assert isinstance(fm_result["equalized_odds"]["passes_equalized_odds_01"], bool)

    def test_tpr_gap_is_non_negative(self, fm_result):
        assert fm_result["equalized_odds"]["tpr_gap"] >= 0

    def test_fpr_gap_is_non_negative(self, fm_result):
        assert fm_result["equalized_odds"]["fpr_gap"] >= 0

    def test_promotion_rates_are_percentages(self, fm_result):
        assert 0 <= fm_result["predicted_promotion_rate_male"] <= 100
        assert 0 <= fm_result["predicted_promotion_rate_female"] <= 100
