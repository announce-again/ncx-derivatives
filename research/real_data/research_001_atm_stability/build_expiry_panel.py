from __future__ import annotations

from collections import defaultdict
from statistics import median

from ncx_derivatives.market_data import OptionType
from ncx_derivatives.volatility import (
    ImpliedVolatilityChain,
    SmileIvSource,
    SmileSelectionConfig,
    analyze_volatility_smiles,
    build_volatility_smiles,
)
from research.real_data.common._optional import require_pandas


def build_expiry_panel(
    chain: ImpliedVolatilityChain,
    *,
    dataset_id: str,
    minimum_smile_points: int = 5,
    selection_config: SmileSelectionConfig | None = None,
):
    """Build one auditable expiry table from the public Stage 3.1/3.2 APIs."""

    pandas = require_pandas()
    if not dataset_id:
        raise ValueError("dataset_id must not be empty")
    if minimum_smile_points < 2:
        raise ValueError("minimum_smile_points must be at least 2")
    base_config = selection_config or SmileSelectionConfig()
    source_results = {}
    for source in (SmileIvSource.BID, SmileIvSource.MIDPOINT, SmileIvSource.ASK):
        config = SmileSelectionConfig(
            iv_source=source,
            otm_only=base_config.otm_only,
            require_two_sided_quote=base_config.require_two_sided_quote,
            duplicate_strike_policy=base_config.duplicate_strike_policy,
            excluded_diagnostic_flags=base_config.excluded_diagnostic_flags,
            atm_log_moneyness_tolerance=base_config.atm_log_moneyness_tolerance,
            max_relative_spread=base_config.max_relative_spread,
            min_bid_size=base_config.min_bid_size,
            min_ask_size=base_config.min_ask_size,
            min_session_volume=base_config.min_session_volume,
            min_open_interest=base_config.min_open_interest,
            market_state_relative_tolerance=base_config.market_state_relative_tolerance,
            market_state_absolute_tolerance=base_config.market_state_absolute_tolerance,
        )
        selection = build_volatility_smiles(chain, config)
        analysis = analyze_volatility_smiles(selection.smiles)
        source_results[source] = (selection, analysis)

    keys = sorted(
        {
            result.smile.sort_key
            for _, analysis in source_results.values()
            for result in analysis.local_metrics
        }
    )
    rows = []
    attrition = []
    maps = {
        source: {
            metric.smile.sort_key: (metric, delta)
            for metric, delta in zip(analysis.local_metrics, analysis.delta_metrics)
        }
        for source, (_, analysis) in source_results.items()
    }
    excluded = {
        source: _excluded_counts(selection)
        for source, (selection, _) in source_results.items()
    }
    for source, (selection, _) in source_results.items():
        for diagnostic in selection.group_diagnostics:
            attrition.append(
                {
                    "stage": "smile_selection",
                    "reason": f"{source.value}:{diagnostic.reason.value}",
                    "underlying": diagnostic.underlying_symbol,
                    "quote_date": diagnostic.valuation_timestamp.date(),
                    "expiration": diagnostic.expiration,
                    "raw_count": diagnostic.quote_count,
                    "remaining_count": 0,
                }
            )
    for key in keys:
        midpoint_pair = maps[SmileIvSource.MIDPOINT].get(key)
        representative = midpoint_pair or next(
            value.get(key) for value in maps.values() if value.get(key) is not None
        )
        representative_metric, _ = representative
        midpoint_metric, midpoint_delta = (
            midpoint_pair if midpoint_pair is not None else (None, None)
        )
        smile = representative_metric.smile
        first = smile.points[0] if smile.points else None
        row = {
            "dataset_id": dataset_id,
            "underlying": smile.underlying_symbol,
            "quote_date": smile.valuation_timestamp.date(),
            "valuation_timestamp": smile.valuation_timestamp,
            "expiration": smile.expiration,
            "actual_dte": round(smile.time_to_maturity * 365),
            "spot": smile.spot_price,
            "forward": smile.forward_price,
            "risk_free_discount_factor": _enriched(first, "risk_free_discount_factor"),
            "dividend_discount_factor": _enriched(first, "dividend_discount_factor"),
        }
        for source, output_name in (
            (SmileIvSource.BID, "bid"),
            (SmileIvSource.MIDPOINT, "mid"),
            (SmileIvSource.ASK, "ask"),
        ):
            pair = maps[source].get(key)
            if pair is None:
                row[f"atm_{output_name}_iv"] = None
                row[f"atm_{output_name}_status"] = "MISSING_SMILE"
                continue
            metric, _ = pair
            enough = len(metric.smile.points) >= minimum_smile_points
            row[f"atm_{output_name}_iv"] = metric.atm.atm_volatility if enough and metric.atm.is_success else None
            row[f"atm_{output_name}_status"] = (
                metric.atm.status.value if enough else "INSUFFICIENT_RESEARCH_POINTS"
            )
            if not enough or not metric.atm.is_success:
                attrition.append(
                    _attrition_row(metric, source, minimum_smile_points)
                )
        row["atm_total_variance"] = (
            row["atm_mid_iv"] ** 2 * smile.time_to_maturity
            if row["atm_mid_iv"] is not None
            else None
        )
        row["atm_method"] = (
            midpoint_metric.atm.method.value
            if midpoint_metric is not None and midpoint_metric.atm.method
            else None
        )
        row["nearest_left_k"] = _point_k(midpoint_metric.atm.left_point) if midpoint_metric else None
        row["nearest_right_k"] = _point_k(midpoint_metric.atm.right_point) if midpoint_metric else None
        row["atm_bracket_span"] = _bracket_span(midpoint_metric) if midpoint_metric else None
        row["selected_point_count"] = len(midpoint_metric.smile.points) if midpoint_metric else 0
        row["excluded_quote_count"] = excluded[SmileIvSource.MIDPOINT].get(key, 0)
        _add_liquidity(row, midpoint_metric.smile.points if midpoint_metric else ())
        row["local_skew"] = midpoint_metric.skew.total_variance_skew_slope if midpoint_metric else None
        row["curvature"] = midpoint_metric.curvature.total_variance_curvature if midpoint_metric else None
        call25 = midpoint_delta.delta_result(OptionType.CALL, 0.25) if midpoint_delta else None
        put25 = midpoint_delta.delta_result(OptionType.PUT, 0.25) if midpoint_delta else None
        rr25 = midpoint_delta.risk_reversal(0.25) if midpoint_delta else None
        bf25 = midpoint_delta.butterfly(0.25) if midpoint_delta else None
        row["call_25d_iv"] = call25.implied_volatility if call25 and call25.is_success else None
        row["put_25d_iv"] = put25.implied_volatility if put25 and put25.is_success else None
        row["rr25"] = rr25.value if rr25 and rr25.is_success else None
        row["bf25"] = bf25.value if bf25 and bf25.is_success else None
        rows.append(row)
    panel = pandas.DataFrame.from_records(rows).sort_values(
        ["underlying", "quote_date", "expiration"], kind="mergesort", ignore_index=True
    ) if rows else pandas.DataFrame()
    attrition_frame = pandas.DataFrame.from_records(attrition)
    if not attrition_frame.empty:
        attrition_frame = attrition_frame.sort_values(
            ["stage", "reason", "underlying", "quote_date", "expiration"],
            kind="mergesort", ignore_index=True,
        )
    return panel, attrition_frame


def _excluded_counts(selection):
    counts = defaultdict(int)
    for diagnostic in selection.diagnostics:
        enriched = diagnostic.iv_quote.enriched_quote
        contract = enriched.quote.contract
        counts[(contract.underlying_symbol, enriched.valuation_timestamp, contract.expiration)] += 1
    for diagnostic in selection.group_diagnostics:
        counts[diagnostic.sort_key] += diagnostic.quote_count
    return counts


def _attrition_row(metric, source, minimum):
    reason = (
        f"selected_points_below_{minimum}"
        if len(metric.smile.points) < minimum
        else metric.atm.failure_reason.value
    )
    return {
        "stage": "atm_metric",
        "reason": f"{source.value}:{reason}",
        "underlying": metric.smile.underlying_symbol,
        "quote_date": metric.smile.valuation_timestamp.date(),
        "expiration": metric.smile.expiration,
        "raw_count": len(metric.smile.points),
        "remaining_count": 0,
    }


def _enriched(point, name):
    return None if point is None else getattr(point.iv_quote.enriched_quote, name)


def _point_k(point):
    return None if point is None else point.log_forward_moneyness


def _bracket_span(metric):
    left = _point_k(metric.atm.left_point)
    right = _point_k(metric.atm.right_point)
    return None if left is None or right is None else right - left


def _add_liquidity(row, points):
    enriched = [point.iv_quote.enriched_quote for point in points]
    price_spreads = [item.absolute_spread for item in enriched if item.absolute_spread is not None]
    relative = [item.relative_spread for item in enriched if item.relative_spread is not None]
    vegas = [point.vega for point in points if point.vega is not None]
    row["median_price_spread"] = median(price_spreads) if price_spreads else None
    row["median_relative_price_spread"] = median(relative) if relative else None
    row["total_volume"] = sum(item.quote.session_volume or 0 for item in enriched)
    row["total_open_interest"] = sum(item.quote.open_interest or 0 for item in enriched)
    row["median_vega"] = median(vegas) if vegas else None
