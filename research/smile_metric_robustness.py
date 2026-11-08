"""Deterministic end-to-end robustness research for Stage 3.1 and Stage 3.2.

The synthetic truth is quadratic in log-forward-moneyness ``k``::

    w(k) = level + slope * k + 0.5 * curvature * k**2

Each experiment prices an OTM option at every retained strike, adds Gaussian
noise to the option-price midpoint, and runs the public production pipeline.
This module deliberately lives outside ``src/ncx_derivatives``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from math import exp, isfinite, sqrt
from pathlib import Path
from statistics import fmean

from ncx_derivatives.greeks import call_delta, put_delta
from ncx_derivatives.market_data import (
    CarryAssumptions,
    CleaningConfig,
    FlatDividendYieldCurve,
    FlatZeroRateCurve,
)
from ncx_derivatives.pricing import call_price, put_price
from ncx_derivatives.volatility import (
    SYNTHETIC_OPTION_QUOTE_COLUMNS,
    build_volatility_smiles,
    calculate_smile_delta_metrics_for_smiles,
    calculate_smile_metrics_for_smiles,
    run_csv_volatility_pipeline,
    synthetic_option_quote_csv_config,
)

METRICS = ("atm", "skew", "curvature", "rr25", "bf25")
EXPERIMENT_COLUMNS = (
    "experiment_id",
    "seed",
    "replicate",
    "price_noise_std",
    "strike_density",
    "input_quote_count",
    "ingested_quote_count",
    "cleaned_quote_count",
    "iv_quote_count",
    "smile_count",
    "selected_point_count",
    "excluded_point_count",
    "atm_truth",
    "atm_recovered",
    "atm_error",
    "atm_success",
    "atm_failure_reason",
    "atm_total_variance_truth",
    "atm_total_variance_recovered",
    "atm_total_variance_error",
    "skew_truth",
    "skew_recovered",
    "skew_error",
    "skew_success",
    "skew_failure_reason",
    "curvature_truth",
    "curvature_recovered",
    "curvature_error",
    "curvature_success",
    "curvature_failure_reason",
    "rr25_truth",
    "rr25_recovered",
    "rr25_error",
    "rr25_success",
    "rr25_failure_reason",
    "bf25_truth",
    "bf25_recovered",
    "bf25_error",
    "bf25_success",
    "bf25_failure_reason",
)
QUOTE_RECORD_COLUMNS = (
    "experiment_id",
    "seed",
    "replicate",
    "price_noise_std",
    "strike_density",
    "strike_index",
    "log_forward_moneyness",
    "strike",
    "option_type",
    "total_variance_truth",
    "volatility_truth",
    "fair_price",
    "price_noise",
    "noisy_midpoint",
    "bid",
    "ask",
)
SUMMARY_COLUMNS = (
    "price_noise_std",
    "strike_density",
    "metric",
    "experiment_count",
    "success_count",
    "success_rate",
    "bias",
    "mae",
    "rmse",
    "selected_point_count_mean",
    "selected_point_count_min",
    "selected_point_count_max",
)


@dataclass(frozen=True, slots=True)
class RobustnessConfig:
    """Configuration for a full factorial noise-by-density experiment."""

    repetitions: int = 100
    base_seed: int = 20260805
    price_noise_stds: tuple[float, ...] = (0.0, 0.005, 0.02)
    strike_densities: tuple[int, ...] = (9, 17, 33)
    min_log_moneyness: float = -0.35
    max_log_moneyness: float = 0.35
    total_variance_level: float = 0.025
    total_variance_slope: float = -0.01
    total_variance_curvature: float = 0.12
    spot: float = 100.0
    expiry_days: int = 182
    risk_free_rate: float = 0.03
    dividend_yield: float = 0.01
    quote_half_spread: float = 0.001
    valuation_timestamp: datetime = datetime(
        2026,
        7,
        30,
        14,
        30,
        tzinfo=timezone.utc,
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.repetitions, bool)
            or not isinstance(self.repetitions, int)
            or self.repetitions <= 0
        ):
            raise ValueError("repetitions must be a positive integer")
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int):
            raise ValueError("base_seed must be an integer")
        noise = tuple(sorted({float(value) for value in self.price_noise_stds}))
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.strike_densities
        ):
            raise ValueError("strike_densities must contain integers")
        densities = tuple(sorted(set(self.strike_densities)))
        if not noise or any(not isfinite(value) or value < 0.0 for value in noise):
            raise ValueError("price_noise_stds must contain non-negative values")
        if not densities or any(value < 3 or value % 2 == 0 for value in densities):
            raise ValueError("strike_densities must contain odd integers >= 3")
        if not self.min_log_moneyness < 0.0 < self.max_log_moneyness:
            raise ValueError("log-moneyness range must straddle zero")
        if self.spot <= 0.0 or self.expiry_days <= 0:
            raise ValueError("spot and expiry_days must be positive")
        if self.quote_half_spread < 0.0:
            raise ValueError("quote_half_spread must be non-negative")
        if self.valuation_timestamp.tzinfo is None:
            raise ValueError("valuation_timestamp must be timezone-aware")
        coordinates = [self.min_log_moneyness, 0.0, self.max_log_moneyness]
        if self.total_variance_curvature > 0.0:
            vertex = -self.total_variance_slope / self.total_variance_curvature
            if self.min_log_moneyness < vertex < self.max_log_moneyness:
                coordinates.append(vertex)
        if any(self.total_variance(k) <= 0.0 for k in coordinates):
            raise ValueError("total variance must be positive across the strike range")
        object.__setattr__(self, "price_noise_stds", noise)
        object.__setattr__(self, "strike_densities", densities)

    @property
    def valuation_date(self) -> date:
        return self.valuation_timestamp.date()

    @property
    def expiration(self) -> date:
        return self.valuation_date + timedelta(days=self.expiry_days)

    @property
    def maturity(self) -> float:
        return self.expiry_days / 365.0

    @property
    def forward(self) -> float:
        return self.spot * exp(
            (self.risk_free_rate - self.dividend_yield) * self.maturity,
        )

    def total_variance(self, log_forward_moneyness: float) -> float:
        k = log_forward_moneyness
        return (
            self.total_variance_level
            + self.total_variance_slope * k
            + 0.5 * self.total_variance_curvature * k * k
        )

    def volatility(self, log_forward_moneyness: float) -> float:
        return sqrt(self.total_variance(log_forward_moneyness) / self.maturity)


@dataclass(frozen=True, slots=True)
class ResearchOutputs:
    output_directory: Path
    experiment_records_path: Path
    quote_records_path: Path
    summary_path: Path
    config_path: Path
    experiment_count: int
    quote_record_count: int


def run_robustness_research(
    output_directory: str | Path,
    config: RobustnessConfig = RobustnessConfig(),
) -> ResearchOutputs:
    """Run all conditions, preserve raw records, and write aggregate statistics."""

    output = Path(output_directory)
    input_directory = output / "stage_3_1_inputs"
    input_directory.mkdir(parents=True, exist_ok=True)
    truths = _metric_truth(config)
    experiments: list[dict[str, object]] = []
    quote_records: list[dict[str, object]] = []

    experiment_number = 0
    for noise in config.price_noise_stds:
        for density in config.strike_densities:
            for replicate in range(config.repetitions):
                experiment_number += 1
                experiment_id = f"experiment_{experiment_number:06d}"
                seed = _experiment_seed(config.base_seed, noise, density, replicate)
                source_path = input_directory / f"{experiment_id}_seed_{seed}.csv"
                records = _write_experiment_quotes(
                    source_path,
                    config,
                    experiment_id=experiment_id,
                    seed=seed,
                    replicate=replicate,
                    price_noise_std=noise,
                    strike_density=density,
                )
                quote_records.extend(records)
                experiments.append(
                    _run_pipeline_experiment(
                        source_path,
                        config,
                        experiment_id=experiment_id,
                        seed=seed,
                        replicate=replicate,
                        price_noise_std=noise,
                        strike_density=density,
                        truths=truths,
                    ),
                )

    summaries = summarize_experiments(experiments)
    experiment_path = output / "experiment_records.csv"
    quote_path = output / "quote_records.csv"
    summary_path = output / "summary.csv"
    config_path = output / "config.json"
    _write_csv(experiment_path, EXPERIMENT_COLUMNS, experiments)
    _write_csv(quote_path, QUOTE_RECORD_COLUMNS, quote_records)
    _write_csv(summary_path, SUMMARY_COLUMNS, summaries)
    _write_config(config_path, config, truths)
    return ResearchOutputs(
        output_directory=output,
        experiment_records_path=experiment_path,
        quote_records_path=quote_path,
        summary_path=summary_path,
        config_path=config_path,
        experiment_count=len(experiments),
        quote_record_count=len(quote_records),
    )


def summarize_experiments(
    experiment_records: Iterable[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    """Calculate condition-level error statistics from raw experiment records."""

    groups: dict[tuple[float, int], list[dict[str, object]]] = {}
    for record in experiment_records:
        key = (float(record["price_noise_std"]), int(record["strike_density"]))
        groups.setdefault(key, []).append(record)

    output: list[dict[str, object]] = []
    for (noise, density), records in sorted(groups.items()):
        selected = [int(record["selected_point_count"]) for record in records]
        for metric in METRICS:
            successes = [record for record in records if record[f"{metric}_success"]]
            errors = [float(record[f"{metric}_error"]) for record in successes]
            output.append(
                {
                    "price_noise_std": noise,
                    "strike_density": density,
                    "metric": metric,
                    "experiment_count": len(records),
                    "success_count": len(successes),
                    "success_rate": len(successes) / len(records),
                    "bias": _mean_or_none(errors),
                    "mae": _mean_or_none([abs(value) for value in errors]),
                    "rmse": (
                        None
                        if not errors
                        else sqrt(fmean(value * value for value in errors))
                    ),
                    "selected_point_count_mean": fmean(selected),
                    "selected_point_count_min": min(selected),
                    "selected_point_count_max": max(selected),
                },
            )
    return tuple(output)


def _run_pipeline_experiment(
    source_path: Path,
    config: RobustnessConfig,
    *,
    experiment_id: str,
    seed: int,
    replicate: int,
    price_noise_std: float,
    strike_density: int,
    truths: dict[str, float],
) -> dict[str, object]:
    carry = CarryAssumptions(
        risk_free_curve=FlatZeroRateCurve(config.risk_free_rate),
        dividend_curve=FlatDividendYieldCurve(config.dividend_yield),
    )
    pipeline = run_csv_volatility_pipeline(
        source_path,
        ingestion_config=synthetic_option_quote_csv_config(),
        carry=carry,
        valuation_date=config.valuation_date,
        cleaning_config=CleaningConfig(
            reject_missing_bid=True,
            reject_missing_ask=True,
            reject_crossed_market=True,
            reject_zero_midpoint=True,
        ),
    )
    selection = build_volatility_smiles(pipeline.implied_volatility_chain)
    local_metrics = calculate_smile_metrics_for_smiles(selection.smiles)
    delta_metrics = calculate_smile_delta_metrics_for_smiles(
        selection.smiles,
        local_metric_results=local_metrics,
    )

    recovered: dict[str, float | None] = {metric: None for metric in METRICS}
    failures: dict[str, str | None] = {metric: "NO_SMILE" for metric in METRICS}
    atm_variance: float | None = None
    if local_metrics:
        local = local_metrics[0]
        recovered.update(
            atm=local.atm.atm_volatility,
            skew=local.skew.total_variance_skew_slope,
            curvature=local.curvature.total_variance_curvature,
        )
        atm_variance = local.atm.atm_total_variance
        failures.update(
            atm=_reason(local.atm.failure_reason),
            skew=_reason(local.skew.failure_reason),
            curvature=_reason(local.curvature.failure_reason),
        )
    if delta_metrics:
        delta = delta_metrics[0]
        rr = delta.risk_reversal(0.25)
        bf = delta.butterfly(0.25)
        if rr is not None:
            recovered["rr25"] = rr.value
            failures["rr25"] = _reason(rr.failure_reason)
        if bf is not None:
            recovered["bf25"] = bf.value
            failures["bf25"] = _reason(bf.failure_reason)

    counts = pipeline.counts
    record: dict[str, object] = {
        "experiment_id": experiment_id,
        "seed": seed,
        "replicate": replicate,
        "price_noise_std": price_noise_std,
        "strike_density": strike_density,
        "input_quote_count": counts.input_row_count,
        "ingested_quote_count": counts.ingestion_success_count,
        "cleaned_quote_count": counts.cleaning_accepted_count,
        "iv_quote_count": counts.iv_quote_count,
        "smile_count": len(selection.smiles),
        "selected_point_count": selection.summary.selected_point_count,
        "excluded_point_count": selection.summary.excluded_quote_count,
    }
    for metric in METRICS:
        value = recovered[metric]
        record[f"{metric}_truth"] = truths[metric]
        record[f"{metric}_recovered"] = value
        record[f"{metric}_error"] = (
            None if value is None else value - truths[metric]
        )
        record[f"{metric}_success"] = value is not None
        record[f"{metric}_failure_reason"] = failures[metric]
    record["atm_total_variance_truth"] = config.total_variance_level
    record["atm_total_variance_recovered"] = atm_variance
    record["atm_total_variance_error"] = (
        None
        if atm_variance is None
        else atm_variance - config.total_variance_level
    )
    return record


def _write_experiment_quotes(
    path: Path,
    config: RobustnessConfig,
    *,
    experiment_id: str,
    seed: int,
    replicate: int,
    price_noise_std: float,
    strike_density: int,
) -> tuple[dict[str, object], ...]:
    rng = random.Random(seed)
    coordinates = _strike_coordinates(
        strike_density,
        config.min_log_moneyness,
        config.max_log_moneyness,
    )
    rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for strike_index, k in enumerate(coordinates):
        strike = config.forward * exp(k)
        option_type = "put" if k < 0.0 else "call"
        volatility = config.volatility(k)
        price_function = put_price if option_type == "put" else call_price
        fair_price = price_function(
            config.spot,
            strike,
            config.maturity,
            config.risk_free_rate,
            volatility,
            config.dividend_yield,
        )
        noise = rng.gauss(0.0, price_noise_std) if price_noise_std else 0.0
        midpoint = max(fair_price + noise, 1e-10)
        half_spread = min(config.quote_half_spread, 0.49 * midpoint)
        bid = midpoint - half_spread
        ask = midpoint + half_spread
        rows.append(
            _quote_row(
                config,
                strike_index=strike_index,
                strike=strike,
                option_type=option_type,
                bid=bid,
                ask=ask,
            ),
        )
        records.append(
            {
                "experiment_id": experiment_id,
                "seed": seed,
                "replicate": replicate,
                "price_noise_std": price_noise_std,
                "strike_density": strike_density,
                "strike_index": strike_index,
                "log_forward_moneyness": k,
                "strike": strike,
                "option_type": option_type,
                "total_variance_truth": config.total_variance(k),
                "volatility_truth": volatility,
                "fair_price": fair_price,
                "price_noise": noise,
                "noisy_midpoint": midpoint,
                "bid": bid,
                "ask": ask,
            },
        )
    _write_csv(path, SYNTHETIC_OPTION_QUOTE_COLUMNS, rows)
    return tuple(records)


def _quote_row(
    config: RobustnessConfig,
    *,
    strike_index: int,
    strike: float,
    option_type: str,
    bid: float,
    ask: float,
) -> dict[str, object]:
    timestamp = config.valuation_timestamp.isoformat()
    return {
        "Underlying Symbol": "ROBUST",
        "Expiration": config.expiration.isoformat(),
        "Strike": _format_float(strike),
        "Option Type": option_type,
        "Quote Datetime": timestamp,
        "Snapshot Datetime": timestamp,
        "Exercise Style": "european",
        "Contract Multiplier": 100,
        "Currency": "USD",
        "Source Contract ID": f"ROBUST-{strike_index:04d}-{option_type[0].upper()}",
        "Bid": _format_float(bid),
        "Ask": _format_float(ask),
        "Bid Size": 100,
        "Ask Size": 100,
        "Session Volume": 1000,
        "Open Interest": 1000,
        "Open Interest Date": (config.valuation_date - timedelta(days=1)).isoformat(),
        "Active Underlying Price": _format_float(config.spot),
        "Underlying Bid": _format_float(config.spot - 0.01),
        "Underlying Ask": _format_float(config.spot + 0.01),
        "Underlying Datetime": timestamp,
    }


def _metric_truth(config: RobustnessConfig) -> dict[str, float]:
    atm = config.volatility(0.0)
    call_k = _solve_delta_coordinate(config, "call", 0.25)
    put_k = _solve_delta_coordinate(config, "put", -0.25)
    call_volatility = config.volatility(call_k)
    put_volatility = config.volatility(put_k)
    return {
        "atm": atm,
        "skew": config.total_variance_slope,
        "curvature": config.total_variance_curvature,
        "rr25": call_volatility - put_volatility,
        "bf25": 0.5 * (call_volatility + put_volatility) - atm,
    }


def _solve_delta_coordinate(
    config: RobustnessConfig,
    option_type: str,
    target: float,
) -> float:
    delta_function = call_delta if option_type == "call" else put_delta

    def residual(k: float) -> float:
        return delta_function(
            config.spot,
            config.forward * exp(k),
            config.maturity,
            config.risk_free_rate,
            config.volatility(k),
            config.dividend_yield,
        ) - target

    start, stop = (
        (0.0, config.max_log_moneyness)
        if option_type == "call"
        else (config.min_log_moneyness, 0.0)
    )
    grid = [start + (stop - start) * index / 1000 for index in range(1001)]
    brackets = [
        (left, right)
        for left, right in zip(grid, grid[1:])
        if residual(left) == 0.0 or residual(left) * residual(right) <= 0.0
    ]
    if not brackets:
        raise ValueError(f"continuous truth does not bracket {target:+.2f} delta")
    left, right = min(brackets, key=lambda pair: abs(pair[0]) + abs(pair[1]))
    left_value = residual(left)
    for _ in range(100):
        midpoint = 0.5 * (left + right)
        mid_value = residual(midpoint)
        if abs(mid_value) <= 1e-14:
            return midpoint
        if left_value * mid_value <= 0.0:
            right = midpoint
        else:
            left = midpoint
            left_value = mid_value
    return 0.5 * (left + right)


def _strike_coordinates(count: int, minimum: float, maximum: float) -> tuple[float, ...]:
    step = (maximum - minimum) / (count - 1)
    values = [minimum + index * step for index in range(count)]
    values[count // 2] = 0.0
    return tuple(values)


def _experiment_seed(
    base_seed: int,
    noise: float,
    density: int,
    replicate: int,
) -> int:
    payload = f"{base_seed}|{noise.hex()}|{density}|{replicate}".encode("ascii")
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _reason(reason: object | None) -> str | None:
    if reason is None:
        return None
    return str(getattr(reason, "value", reason))


def _mean_or_none(values: Sequence[float]) -> float | None:
    return None if not values else fmean(values)


def _format_float(value: float) -> str:
    return f"{value:.17g}"


def _write_csv(
    path: Path,
    columns: Sequence[str],
    records: Iterable[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {column: _csv_value(record.get(column)) for column in columns}
            for record in records
        )


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _write_config(
    path: Path,
    config: RobustnessConfig,
    truths: dict[str, float],
) -> None:
    values = asdict(config)
    values["valuation_timestamp"] = config.valuation_timestamp.isoformat()
    payload = {
        "config": values,
        "metric_truth": truths,
        "metric_units": {
            "atm": "annualized volatility",
            "skew": "d(total variance) / d(log-forward-moneyness)",
            "curvature": "d2(total variance) / d(log-forward-moneyness)2",
            "rr25": "annualized volatility",
            "bf25": "annualized volatility",
        },
        "price_noise_units": "option price currency units (Gaussian standard deviation)",
        "strike_density_definition": "number of evenly spaced OTM smile points",
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".tmp/research/smile_metric_robustness"),
    )
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--noise", default="0,0.005,0.02")
    parser.add_argument("--densities", default="9,17,33")
    parser.add_argument("--level", type=float, default=0.025)
    parser.add_argument("--slope", type=float, default=-0.01)
    parser.add_argument("--curvature", type=float, default=0.12)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    config = RobustnessConfig(
        repetitions=arguments.repetitions,
        base_seed=arguments.seed,
        price_noise_stds=_comma_floats(arguments.noise),
        strike_densities=_comma_ints(arguments.densities),
        total_variance_level=arguments.level,
        total_variance_slope=arguments.slope,
        total_variance_curvature=arguments.curvature,
    )
    outputs = run_robustness_research(arguments.output_dir, config)
    print(
        f"Completed {outputs.experiment_count} experiments and "
        f"preserved {outputs.quote_record_count} quote records."
    )
    print(f"Experiment records: {outputs.experiment_records_path}")
    print(f"Condition summary: {outputs.summary_path}")
    print(f"Configuration and truth: {outputs.config_path}")


def _comma_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


if __name__ == "__main__":
    main()
