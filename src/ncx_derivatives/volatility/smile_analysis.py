from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ncx_derivatives.market_data import OptionType
from ncx_derivatives.utils._files import csv_cell, file_sha256
from ncx_derivatives.utils._optional import require_pandas

from .delta_metrics import (
    BUTTERFLY_COLUMNS,
    DELTA_STRUCTURE_COLUMNS,
    DELTA_VOLATILITY_COLUMNS,
    RISK_REVERSAL_COLUMNS,
    DeltaMetricConfig,
    SmileDeltaMetrics,
    butterfly_results_to_records,
    calculate_smile_delta_metrics_for_smiles,
    delta_structure_results_to_records,
    delta_volatility_results_to_records,
    risk_reversal_results_to_records,
)
from .smile_metrics import (
    SMILE_METRIC_COLUMNS,
    SmileMetricConfig,
    SmileMetricResult,
    SmileMetricStatus,
    calculate_smile_metrics_for_smiles,
    smile_metrics_to_records,
)
from .smiles import VolatilitySmile
from .term_structure import (
    TERM_STRUCTURE_COLUMNS,
    VolatilityTermStructure,
    build_volatility_term_structures,
    volatility_term_structures_to_records,
)


class SmileAnalysisMetric(str, Enum):
    ATM = "ATM"
    SKEW = "SKEW"
    CURVATURE = "CURVATURE"
    DELTA_CALL = "DELTA_CALL"
    DELTA_PUT = "DELTA_PUT"
    RISK_REVERSAL = "RISK_REVERSAL"
    BUTTERFLY = "BUTTERFLY"


@dataclass(frozen=True, slots=True)
class SmileAnalysisConfig:
    local_metric_config: SmileMetricConfig = field(
        default_factory=SmileMetricConfig,
    )
    delta_metric_config: DeltaMetricConfig = field(
        default_factory=DeltaMetricConfig,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.local_metric_config, SmileMetricConfig):
            raise ValueError("local_metric_config must be a SmileMetricConfig")
        if not isinstance(self.delta_metric_config, DeltaMetricConfig):
            raise ValueError("delta_metric_config must be a DeltaMetricConfig")


@dataclass(frozen=True, slots=True)
class SmileAnalysisOutcomeCount:
    metric: SmileAnalysisMetric
    delta_magnitude: float | None
    success_count: int
    failure_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.metric, SmileAnalysisMetric):
            raise ValueError("metric must be a SmileAnalysisMetric")
        if self.metric in {
            SmileAnalysisMetric.ATM,
            SmileAnalysisMetric.SKEW,
            SmileAnalysisMetric.CURVATURE,
        }:
            if self.delta_magnitude is not None:
                raise ValueError("local metric counts cannot have delta_magnitude")
        elif (
            isinstance(self.delta_magnitude, bool)
            or not isinstance(self.delta_magnitude, (int, float))
            or not 0.0 < self.delta_magnitude < 1.0
        ):
            raise ValueError("delta metric counts require a valid magnitude")
        for value in (self.success_count, self.failure_count):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("counts must be non-negative integers")


@dataclass(frozen=True, slots=True)
class SmileAnalysisSummary:
    input_smile_count: int
    local_metric_result_count: int
    delta_metric_result_count: int
    term_structure_count: int
    term_structure_point_count: int
    outcome_counts: tuple[SmileAnalysisOutcomeCount, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "input_smile_count",
            "local_metric_result_count",
            "delta_metric_result_count",
            "term_structure_count",
            "term_structure_point_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        counts = tuple(self.outcome_counts)
        if any(not isinstance(item, SmileAnalysisOutcomeCount) for item in counts):
            raise ValueError("outcome_counts must contain outcome count objects")
        if not (
            self.input_smile_count
            == self.local_metric_result_count
            == self.delta_metric_result_count
            == self.term_structure_point_count
        ):
            raise ValueError("summary result counts must conserve input smiles")
        if any(
            item.success_count + item.failure_count != self.input_smile_count
            for item in counts
        ):
            raise ValueError("each outcome count must conserve input smiles")
        object.__setattr__(self, "outcome_counts", counts)

    def outcome(
        self,
        metric: SmileAnalysisMetric,
        delta_magnitude: float | None = None,
    ) -> SmileAnalysisOutcomeCount | None:
        return next(
            (
                count
                for count in self.outcome_counts
                if count.metric is metric
                and count.delta_magnitude == delta_magnitude
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
class SmileAnalysisResult:
    local_metrics: tuple[SmileMetricResult, ...]
    delta_metrics: tuple[SmileDeltaMetrics, ...]
    term_structures: tuple[VolatilityTermStructure, ...]
    config: SmileAnalysisConfig

    def __post_init__(self) -> None:
        if not isinstance(self.config, SmileAnalysisConfig):
            raise ValueError("config must be a SmileAnalysisConfig")
        local_values = tuple(self.local_metrics)
        delta_values = tuple(self.delta_metrics)
        term_values = tuple(self.term_structures)
        if any(not isinstance(item, SmileMetricResult) for item in local_values):
            raise ValueError("local_metrics must contain SmileMetricResult")
        if any(not isinstance(item, SmileDeltaMetrics) for item in delta_values):
            raise ValueError("delta_metrics must contain SmileDeltaMetrics")
        if any(not isinstance(item, VolatilityTermStructure) for item in term_values):
            raise ValueError("term_structures must contain VolatilityTermStructure")
        local = tuple(sorted(local_values, key=lambda item: item.sort_key))
        delta = tuple(sorted(delta_values, key=lambda item: item.sort_key))
        terms = tuple(sorted(term_values, key=lambda item: item.sort_key))
        if [id(item.smile) for item in local] != [
            id(item.smile) for item in delta
        ]:
            raise ValueError("local and delta metrics must reference identical smiles")
        term_smile_ids = sorted(
            id(point.smile)
            for structure in terms
            for point in structure.points
        )
        if term_smile_ids != sorted(id(item.smile) for item in local):
            raise ValueError("term structures must preserve every analyzed smile")
        object.__setattr__(self, "local_metrics", local)
        object.__setattr__(self, "delta_metrics", delta)
        object.__setattr__(self, "term_structures", terms)

    @property
    def summary(self) -> SmileAnalysisSummary:
        counts = []
        for metric, values in (
            (SmileAnalysisMetric.ATM, (item.atm for item in self.local_metrics)),
            (SmileAnalysisMetric.SKEW, (item.skew for item in self.local_metrics)),
            (
                SmileAnalysisMetric.CURVATURE,
                (item.curvature for item in self.local_metrics),
            ),
        ):
            values_tuple = tuple(values)
            counts.append(_outcome_count(metric, None, values_tuple))
        for magnitude in self.config.delta_metric_config.standard_delta_magnitudes:
            calls = tuple(
                item.delta_result(OptionType.CALL, magnitude)
                for item in self.delta_metrics
            )
            puts = tuple(
                item.delta_result(OptionType.PUT, magnitude)
                for item in self.delta_metrics
            )
            reversals = tuple(
                item.risk_reversal(magnitude) for item in self.delta_metrics
            )
            butterflies = tuple(
                item.butterfly(magnitude) for item in self.delta_metrics
            )
            counts.extend(
                (
                    _outcome_count(
                        SmileAnalysisMetric.DELTA_CALL,
                        magnitude,
                        calls,
                    ),
                    _outcome_count(
                        SmileAnalysisMetric.DELTA_PUT,
                        magnitude,
                        puts,
                    ),
                    _outcome_count(
                        SmileAnalysisMetric.RISK_REVERSAL,
                        magnitude,
                        reversals,
                    ),
                    _outcome_count(
                        SmileAnalysisMetric.BUTTERFLY,
                        magnitude,
                        butterflies,
                    ),
                ),
            )
        return SmileAnalysisSummary(
            input_smile_count=len(self.local_metrics),
            local_metric_result_count=len(self.local_metrics),
            delta_metric_result_count=len(self.delta_metrics),
            term_structure_count=len(self.term_structures),
            term_structure_point_count=sum(
                len(structure.points) for structure in self.term_structures
            ),
            outcome_counts=tuple(counts),
        )


@dataclass(frozen=True, slots=True)
class SmileAnalysisCsvExport:
    name: str
    path: Path
    row_count: int
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SmileAnalysisCsvExports:
    exports: tuple[SmileAnalysisCsvExport, ...]

    def by_name(self, name: str) -> SmileAnalysisCsvExport:
        return next(export for export in self.exports if export.name == name)


SMILE_ANALYSIS_SUMMARY_COLUMNS = (
    "input_smile_count",
    "local_metric_result_count",
    "delta_metric_result_count",
    "term_structure_count",
    "term_structure_point_count",
    "metric",
    "delta_magnitude",
    "success_count",
    "failure_count",
)


def analyze_volatility_smiles(
    smiles: Iterable[VolatilitySmile],
    config: SmileAnalysisConfig | None = None,
) -> SmileAnalysisResult:
    analysis_config = config or SmileAnalysisConfig()
    if not isinstance(analysis_config, SmileAnalysisConfig):
        raise ValueError("config must be a SmileAnalysisConfig or None")
    ordered = tuple(sorted(tuple(smiles), key=lambda smile: smile.sort_key))
    if any(not isinstance(smile, VolatilitySmile) for smile in ordered):
        raise ValueError("smiles must contain VolatilitySmile objects")
    local = calculate_smile_metrics_for_smiles(
        ordered,
        analysis_config.local_metric_config,
    )
    delta = calculate_smile_delta_metrics_for_smiles(
        ordered,
        analysis_config.delta_metric_config,
        local_metric_results=local,
    )
    terms = build_volatility_term_structures(local, delta) if ordered else ()
    return SmileAnalysisResult(
        local_metrics=local,
        delta_metrics=delta,
        term_structures=terms,
        config=analysis_config,
    )


def smile_analysis_summary_to_records(
    result: SmileAnalysisResult,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(result, SmileAnalysisResult):
        raise ValueError("result must be a SmileAnalysisResult")
    summary = result.summary
    return tuple(
        {
            "input_smile_count": summary.input_smile_count,
            "local_metric_result_count": summary.local_metric_result_count,
            "delta_metric_result_count": summary.delta_metric_result_count,
            "term_structure_count": summary.term_structure_count,
            "term_structure_point_count": summary.term_structure_point_count,
            "metric": count.metric.value,
            "delta_magnitude": count.delta_magnitude,
            "success_count": count.success_count,
            "failure_count": count.failure_count,
        }
        for count in summary.outcome_counts
    )


def smile_analysis_summary_to_dataframe(result: SmileAnalysisResult):
    pandas = require_pandas()
    return pandas.DataFrame.from_records(
        smile_analysis_summary_to_records(result),
        columns=SMILE_ANALYSIS_SUMMARY_COLUMNS,
    )


def write_smile_analysis_csv(
    directory: str | Path,
    result: SmileAnalysisResult,
) -> SmileAnalysisCsvExports:
    if not isinstance(result, SmileAnalysisResult):
        raise ValueError("result must be a SmileAnalysisResult")
    output = Path(directory)
    datasets = (
        (
            "local_metrics",
            "smile_metrics.csv",
            SMILE_METRIC_COLUMNS,
            smile_metrics_to_records(result.local_metrics),
        ),
        (
            "delta_volatility",
            "delta_volatility.csv",
            DELTA_VOLATILITY_COLUMNS,
            delta_volatility_results_to_records(result.delta_metrics),
        ),
        (
            "risk_reversals",
            "risk_reversals.csv",
            RISK_REVERSAL_COLUMNS,
            risk_reversal_results_to_records(result.delta_metrics),
        ),
        (
            "butterflies",
            "butterflies.csv",
            BUTTERFLY_COLUMNS,
            butterfly_results_to_records(result.delta_metrics),
        ),
        (
            "delta_structures",
            "delta_structures.csv",
            DELTA_STRUCTURE_COLUMNS,
            delta_structure_results_to_records(result.delta_metrics),
        ),
        (
            "term_structures",
            "term_structures.csv",
            TERM_STRUCTURE_COLUMNS,
            volatility_term_structures_to_records(result.term_structures),
        ),
        (
            "summary",
            "analysis_summary.csv",
            SMILE_ANALYSIS_SUMMARY_COLUMNS,
            smile_analysis_summary_to_records(result),
        ),
    )
    exports = tuple(
        _write_records(output / filename, name, columns, records)
        for name, filename, columns, records in datasets
    )
    return SmileAnalysisCsvExports(exports=exports)


def _outcome_count(
    metric: SmileAnalysisMetric,
    delta_magnitude: float | None,
    values: tuple,
) -> SmileAnalysisOutcomeCount:
    success = sum(
        value is not None and value.status is SmileMetricStatus.SUCCESS
        for value in values
    )
    return SmileAnalysisOutcomeCount(
        metric=metric,
        delta_magnitude=delta_magnitude,
        success_count=success,
        failure_count=len(values) - success,
    )


def _write_records(
    path: Path,
    name: str,
    columns: tuple[str, ...],
    records: tuple[dict[str, Any], ...],
) -> SmileAnalysisCsvExport:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {column: csv_cell(record[column]) for column in columns}
            for record in records
        )
    return SmileAnalysisCsvExport(
        name=name,
        path=path,
        row_count=len(records),
        byte_count=path.stat().st_size,
        sha256=file_sha256(path),
    )
