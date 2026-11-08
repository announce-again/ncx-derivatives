import runpy
import shutil
import sys
from pathlib import Path

import pytest

EXAMPLES = (
    "01_build_canonical_snapshot.py",
    "02_ingest_provider_csv.py",
    "03_validate_option_chain.py",
    "04_enrich_option_chain.py",
    "05_clean_option_chain.py",
    "06_diagnose_static_arbitrage.py",
    "07_pandas_interoperability.py",
    "08_write_dataset_snapshot.py",
    "09_end_to_end_pipeline.py",
)
PANDAS_EXAMPLES = {
    "07_pandas_interoperability.py",
    "09_end_to_end_pipeline.py",
}


@pytest.mark.parametrize("filename", EXAMPLES)
def test_market_data_example_runs(filename: str, monkeypatch, tmp_path) -> None:
    if filename in PANDAS_EXAMPLES:
        pytest.importorskip("pandas")
    repo_root = Path(__file__).parents[2]
    output_dir = tmp_path / "examples_output"
    monkeypatch.setenv("NCX_EXAMPLES_OUTPUT_DIR", str(output_dir))
    path = (
        repo_root
        / "examples"
        / "market_data"
        / filename
    )

    try:
        runpy.run_path(str(path), run_name="__main__")
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_stage_3_1_pipeline_example_runs(monkeypatch, tmp_path) -> None:
    repo_root = Path(__file__).parents[2]
    output_dir = tmp_path / "volatility_pipeline"
    path = (
        repo_root
        / "examples"
        / "market_data"
        / "10_stage_3_1_volatility_pipeline.py"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(path), "--rows", "500", "--output-dir", str(output_dir)],
    )

    runpy.run_path(str(path), run_name="__main__")

    assert (output_dir / "synthetic_option_quotes.csv").is_file()
    assert (output_dir / "implied_volatility_chain.csv").is_file()


def test_stage_3_2_analysis_example_runs(monkeypatch, tmp_path) -> None:
    repo_root = Path(__file__).parents[2]
    output_dir = tmp_path / "volatility_smiles"
    path = (
        repo_root
        / "examples"
        / "market_data"
        / "11_stage_3_2_volatility_smiles.py"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(path), "--rows", "500", "--output-dir", str(output_dir)],
    )

    runpy.run_path(str(path), run_name="__main__")

    assert (output_dir / "volatility_smile_points.csv").is_file()
    assert (output_dir / "analysis" / "smile_metrics.csv").is_file()
    assert (output_dir / "analysis" / "delta_volatility.csv").is_file()
    assert (output_dir / "analysis" / "risk_reversals.csv").is_file()
    assert (output_dir / "analysis" / "butterflies.csv").is_file()
    assert (output_dir / "analysis" / "term_structures.csv").is_file()
