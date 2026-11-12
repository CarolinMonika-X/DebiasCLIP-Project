"""Recompute placebo p-values against placebo-run and canonical LEACE references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_NULL = "permuted_race_labels"
METRICS = ("eod_max", "dpd", "fpr_gap")
MISMATCH_TOLERANCE = 0.002


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required results file not found: {path}")
    with path.open(encoding="utf-8") as file:
        result = json.load(file)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return result


def permutation_result(null_values: list[float], reference: float) -> dict[str, Any]:
    n = len(null_values)
    if n == 0:
        raise ValueError("Primary-null distribution is empty.")
    k = sum(value <= reference for value in null_values)
    return {
        "k_null_values_less_than_or_equal_to_reference": k,
        "n": n,
        "p_value": (k + 1) / (n + 1),
        "alternative": "left-tailed; count of null values <= reference",
        "formula": "(k + 1) / (N + 1)",
    }


def build_report(
    placebo_results: dict[str, Any],
    canonical_results: dict[str, Any],
) -> dict[str, Any]:
    if placebo_results.get("primary_null") != PRIMARY_NULL:
        raise ValueError(
            f"Expected primary null {PRIMARY_NULL!r}, "
            f"found {placebo_results.get('primary_null')!r}."
        )

    records = placebo_results["placebo_results"][PRIMARY_NULL]
    placebo_reference = placebo_results["true_leace"]
    canonical_leace = canonical_results["ensemble_metrics"]["leace"]
    n = len(records)

    metric_results: dict[str, Any] = {}
    for metric in METRICS:
        null_values = [float(record[metric]) for record in records]
        placebo_value = float(placebo_reference[metric])
        canonical_key = "eod_max" if metric == "eod_max" else metric
        canonical_value = float(canonical_leace[canonical_key])
        difference = placebo_value - canonical_value
        metric_results[metric] = {
            "placebo_v2_true_leace_reference": placebo_value,
            "canonical_leace_reference": canonical_value,
            "difference_placebo_minus_canonical": difference,
            "absolute_difference": abs(difference),
            "mismatch_tolerance": MISMATCH_TOLERANCE,
            "mismatch": abs(difference) > MISMATCH_TOLERANCE,
            "primary_null_n": n,
            "p_values": {
                "placebo_v2_reference": permutation_result(
                    null_values, placebo_value
                ),
                "canonical_reference": permutation_result(
                    null_values, canonical_value
                ),
            },
        }

    eod_values = [float(record["eod_max"]) for record in records]
    fpr_values = [float(record["fpr_gap"]) for record in records]
    max_abs_difference = max(
        abs(eod - fpr) for eod, fpr in zip(eod_values, fpr_values)
    )
    identical = max_abs_difference == 0.0
    # EOD_max is max(TPR gap, FPR gap); equality is expected whenever FPR gap >= TPR gap.
    identity_explanation = (
        "In each placebo, EOD_max equals FPR gap because FPR gap is at least "
        "as large as TPR gap, so max(TPR gap, FPR gap) selects FPR gap."
        if identical
        else "At least one placebo has different EOD_max and FPR gap values."
    )

    configuration = placebo_results.get("configuration", {})
    return {
        "primary_null": PRIMARY_NULL,
        "primary_null_description": placebo_results.get(
            "primary_null_description"
        ),
        "true_leace_provenance": {
            "model": "LEACE-Race",
            "ensemble": "Five-seed LEACE-Race diagnostic ensemble",
            "seed_list": configuration.get("seeds"),
            "epochs": configuration.get("epochs"),
            "batch_size": configuration.get("batch_size"),
            "threshold": configuration.get("threshold"),
            "threshold_rule": configuration.get("threshold_rule"),
            "leace_fit_split": configuration.get("leace_fit_split"),
            "results_source": "placebo_v2_results.json -> true_leace",
        },
        "canonical_leace_provenance": {
            "model": "LEACE-Race",
            "canonical_run": canonical_results.get("canonical_run"),
            "canonical_source": canonical_results.get("canonical_source"),
            "canonical_section": canonical_results.get("canonical_section"),
            "seed_list": canonical_results.get("configuration", {}).get(
                "seed_list"
            ),
            "results_source": (
                "canonical_results.json -> ensemble_metrics -> leace"
            ),
        },
        "metric_comparison_and_recomputed_p_values": metric_results,
        "eod_max_vs_fpr_gap": {
            "identical_per_placebo": identical,
            "max_absolute_difference": max_abs_difference,
            "number_of_placebos": n,
            "explanation": identity_explanation,
        },
    }


def print_report(report: dict[str, Any]) -> None:
    provenance = report["true_leace_provenance"]
    print(f"Primary null: {report['primary_null']}")
    print(
        "True LEACE reference source: "
        f"{provenance['results_source']}; model={provenance['model']}; "
        f"ensemble={provenance['ensemble']}; seeds={provenance['seed_list']}."
    )
    print(
        f"Training details: epochs={provenance['epochs']}, "
        f"batch_size={provenance['batch_size']}, "
        f"threshold={provenance['threshold']} ({provenance['threshold_rule']}), "
        f"LEACE fit split={provenance['leace_fit_split']}."
    )
    canonical = report["canonical_leace_provenance"]
    print(
        "Canonical reference: "
        f"{canonical['canonical_run']} from {canonical['canonical_source']}; "
        f"LEACE-Race seeds={canonical['seed_list']}."
    )
    print("\nMetric reference values and recomputed one-sided p-values")
    for metric, result in report["metric_comparison_and_recomputed_p_values"].items():
        flag = "MISMATCH" if result["mismatch"] else "within tolerance"
        print(
            f"{metric}: placebo_v2={result['placebo_v2_true_leace_reference']:.10f}; "
            f"canonical={result['canonical_leace_reference']:.10f}; "
            f"absolute difference={result['absolute_difference']:.10f} "
            f"({flag}, tolerance={result['mismatch_tolerance']:.3f})"
        )
        for reference_name, values in result["p_values"].items():
            print(
                f"  {reference_name}: k="
                f"{values['k_null_values_less_than_or_equal_to_reference']}/"
                f"{values['n']}, p={values['p_value']:.10f}"
            )
    identity = report["eod_max_vs_fpr_gap"]
    print(
        "\nEOD_max and FPR gap identical per placebo: "
        f"{identity['identical_per_placebo']}; "
        f"max absolute difference={identity['max_absolute_difference']:.12g}."
    )
    print(identity["explanation"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <root>/placebo_pvalue_check.json).",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    placebo_results = load_json(root / "placebo_v2_results.json")
    canonical_results = load_json(root / "canonical_results.json")
    report = build_report(placebo_results, canonical_results)

    output = args.output.resolve() if args.output else root / "placebo_pvalue_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_report(report)
    print(f"\nSaved report to {output}")


if __name__ == "__main__":
    main()
