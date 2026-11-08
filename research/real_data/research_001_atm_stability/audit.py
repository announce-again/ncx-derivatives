from __future__ import annotations

import argparse
from pathlib import Path

from research.real_data.common.dataset_audit import (
    audit_standardized_options,
    write_audit_outputs,
)
from research.real_data.common.deterministic_io import write_csv
from research.real_data.common.optionsdx_adapter import (
    STANDARD_COLUMNS,
    read_optionsdx_files,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen OptionsDX/Kaggle files")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("files", nargs="+", type=Path)
    arguments = parser.parse_args()
    standardized = read_optionsdx_files(arguments.files, underlying_symbol=arguments.symbol)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        arguments.output_dir / "standardized_contracts.csv",
        standardized.to_dict("records"),
        columns=STANDARD_COLUMNS,
        sort_by=("underlying_symbol", "quote_date", "expiration", "strike", "option_type", "source_file", "source_row"),
    )
    write_audit_outputs(audit_standardized_options(standardized), arguments.output_dir)


if __name__ == "__main__":
    main()

