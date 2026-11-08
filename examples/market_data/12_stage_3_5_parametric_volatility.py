"""Run Stage 3.5 on deterministic quotes through the normal CSV-to-smile path."""

from __future__ import annotations

import argparse
from pathlib import Path

from ncx_derivatives.volatility import (
    CalibrationWeighting,
    SyntheticOptionDatasetConfig,
    build_volatility_smiles,
    calibrate_ssvi,
    calibrate_svi,
    compare_parametric_holdouts,
    run_csv_volatility_pipeline,
    synthetic_option_quote_csv_config,
    synthetic_volatility_pipeline_carry,
    synthetic_volatility_pipeline_cleaning_config,
    write_synthetic_option_quote_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".tmp/examples_output/parametric_volatility"),
    )
    parser.add_argument(
        "--weighting",
        choices=[mode.value for mode in CalibrationWeighting],
        default="uniform",
    )
    args = parser.parse_args()
    weighting = CalibrationWeighting(args.weighting)
    config = SyntheticOptionDatasetConfig(row_count=500)
    source = write_synthetic_option_quote_csv(
        args.output_dir / "synthetic_quotes.csv", config
    )
    chain = run_csv_volatility_pipeline(
        source.path,
        ingestion_config=synthetic_option_quote_csv_config(),
        carry=synthetic_volatility_pipeline_carry(config),
        valuation_date=config.valuation_date,
        cleaning_config=synthetic_volatility_pipeline_cleaning_config(),
    ).implied_volatility_chain
    smiles = tuple(
        smile
        for smile in build_volatility_smiles(chain).smiles
        if len(smile.points) >= 6
    )
    surface = calibrate_ssvi(smiles, weighting=weighting)
    print(
        f"SSVI converged={surface.converged}, weighted total-variance RMSE={surface.weighted_rmse:.8g}"
    )
    for smile in smiles:
        fit = calibrate_svi(smile, weighting=weighting)
        print(
            f"SVI {smile.expiration} converged={fit.converged}, weighted RMSE={fit.weighted_rmse:.8g}, "
            f"minimum sampled density factor={fit.minimum_sampled_butterfly_factor:.6g}"
        )
    print("Maturity  Nonparametric error     SVI error     SSVI error")
    for row in compare_parametric_holdouts(smiles, weighting=weighting):
        observed = row.observed_total_variance
        print(
            f"{row.maturity:8.4f} {row.nonparametric_total_variance - observed:20.8g} "
            f"{row.svi_total_variance - observed:13.8g} {row.ssvi_total_variance - observed:14.8g}"
        )


if __name__ == "__main__":
    main()
