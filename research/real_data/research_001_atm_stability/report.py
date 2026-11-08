from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from research.real_data.common.deterministic_io import write_json, write_output_hashes

LIMITATIONS = (
    "Kaggle is a distribution channel, not the originating exchange.",
    "Provenance depends materially on uploader descriptions and frozen page snapshots.",
    "Complete historical risk-free and dividend curves may be unavailable.",
    "The vendor IV and Greeks model specification may be incomplete.",
    "Similar schemas do not establish that separate Kaggle datasets are homogeneous.",
    "The claimed EOD snapshot time must still be verified from fields and timestamps.",
    "Split adjustment can differ across files and datasets.",
    "The sample does not represent the full US options market.",
    "The 2020-2022 cross-underlying window overweights pandemic and high-volatility conditions.",
    "The five-underlying cross section is exploratory evidence, not a broad population estimate.",
)


def write_limitations(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    body = "# Research 001 limitations\n\n" + "\n".join(
        f"{index}. {text}" for index, text in enumerate(LIMITATIONS, start=1)
    ) + "\n"
    output.write_text(body, encoding="utf-8", newline="\n")
    return output


def write_research_report(
    path: str | Path,
    *,
    title: str,
    result_files: Mapping[str, str],
    notes: tuple[str, ...] = (),
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "## Results artifacts", ""]
    lines.extend(f"- `{name}`: `{location}`" for name, location in sorted(result_files.items()))
    lines.extend(["", "## Interpretation guardrails", ""])
    lines.extend(f"- {note}" for note in notes)
    lines.extend(
        [
            "",
            "Vendor-IV replication and NCX-reconstructed IV are distinct specifications. ",
            "No result should be interpreted as representative of the full US options market.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return output


def write_run_manifest(
    output_directory: str | Path,
    *,
    config_sha256: str,
    dataset_manifests: Mapping[str, str],
    assumptions: Mapping[str, object],
) -> tuple[Path, Path]:
    root = Path(output_directory)
    manifest = write_json(
        root / "run_manifest.json",
        {
            "research_id": "research_001_atm_stability",
            "config_sha256": config_sha256,
            "dataset_manifests": dict(sorted(dataset_manifests.items())),
            "assumptions": dict(assumptions),
        },
    )
    hashes = write_output_hashes(root, root / "output_hashes.json")
    return manifest, hashes
