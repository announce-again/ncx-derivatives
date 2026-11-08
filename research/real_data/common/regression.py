from __future__ import annotations


def fit_hac_ols(frame, *, formula: str, max_lags: int = 5):
    statsmodels_formula = _import_statsmodels_formula()
    model = statsmodels_formula.ols(formula, data=frame).fit(
        cov_type="HAC", cov_kwds={"maxlags": max_lags}
    )
    return model, regression_result_records(model, model_name="hac_ols")


def fit_clustered_ols(frame, *, formula: str, cluster_column: str):
    statsmodels_formula = _import_statsmodels_formula()
    model = statsmodels_formula.ols(formula, data=frame).fit(
        cov_type="cluster", cov_kwds={"groups": frame[cluster_column]}
    )
    return model, regression_result_records(model, model_name="clustered_ols")


def regression_result_records(model, *, model_name: str) -> tuple[dict[str, object], ...]:
    conf = model.conf_int()
    return tuple(
        {
            "model": model_name,
            "term": term,
            "coefficient": float(model.params[term]),
            "standard_error": float(model.bse[term]),
            "t_statistic": float(model.tvalues[term]),
            "p_value": float(model.pvalues[term]),
            "confidence_interval_low": float(conf.loc[term, 0]),
            "confidence_interval_high": float(conf.loc[term, 1]),
            "observation_count": int(model.nobs),
            "r_squared": float(model.rsquared),
        }
        for term in model.params.index
    )


def _import_statsmodels_formula():
    try:
        import statsmodels.formula.api as formula
    except ImportError as error:
        raise RuntimeError(
            "Research 001 regressions require statsmodels; install the research extra"
        ) from error
    return formula

