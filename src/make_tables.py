"""Regenerate manuscript tables and abstract numbers from saved audit results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required results file not found: {path}")
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def number(value: float, digits: int = 4, signed: bool = False) -> str:
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def table(
    *,
    caption: str,
    label: str,
    alignment: str,
    headings: list[str],
    rows: list[list[str]],
    notes: list[str] | None = None,
) -> str:
    lines = [f"% {note}" for note in notes or []]
    lines.extend(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{alignment}}}",
            r"\toprule",
            " & ".join(headings) + r" \\",
            r"\midrule",
        ]
    )
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def canonical_table(results: dict[str, Any]) -> str:
    metrics = results["ensemble_metrics"]
    labels = {
        "raw": "Raw Dual",
        "leace": "LEACE-Race",
        "inlp": "INLP",
        "parity": "Adversarial parity",
        "labelcond": "Label-conditioned",
        "reweight": "Sample reweighting",
        "thresholds": "Group cutoffs",
    }
    rows = [
        [
            labels[name],
            number(metrics[name]["auc"]),
            number(metrics[name]["acc"]),
            number(metrics[name]["dpd"]),
            number(metrics[name]["eod_max"]),
            number(metrics[name]["w_auc"]),
        ]
        for name in labels
    ]
    return table(
        caption=(
            "Canonical five-seed ensemble results at the recorded fixed threshold. "
            "Worst-group AUC is reported as stored in the canonical results."
        ),
        label="tab:primary",
        alignment="lrrrrr",
        headings=[
            "Method",
            "AUC",
            "Accuracy",
            "DPD",
            r"EOD$_{\max}$",
            "Worst-group AUC",
        ],
        rows=rows,
        notes=[
            f"Canonical run: {results['canonical_run']} ({results['canonical_source']}).",
            "All values are read from canonical_results.json; no metrics are recomputed.",
        ],
    )


def seed_table(results: dict[str, Any]) -> str:
    summary = results["paired_delta_summary"]
    intervals = results["bootstrap"]["intervals"]
    rows: list[list[str]] = []
    for metric, label in (
        ("d_auc", r"$\Delta$ AUC"),
        ("d_eod_max", r"$\Delta$ EOD$_{\max}$"),
        ("d_dpd", r"$\Delta$ DPD"),
        ("d_worst_group_auc", r"$\Delta$ worst-group AUC"),
    ):
        entry = summary[metric]
        ci = intervals[metric]["ci_95_percentile"]
        adjusted = intervals[metric]["bonferroni_adjusted_ci"]
        signs = entry["sign_count"]
        rows.append(
            [
                label,
                f"{number(entry['mean'], signed=True)} $\\pm$ {number(entry['sd'])}",
                f"[{number(entry['min'], signed=True)}, {number(entry['max'], signed=True)}]",
                f"{signs['positive']}/{signs['negative']}/{signs['zero']}",
                f"[{number(ci[0], signed=True)}, {number(ci[1], signed=True)}]",
                f"[{number(adjusted[0], signed=True)}, {number(adjusted[1], signed=True)}]",
            ]
        )
    return table(
        caption=(
            "Paired seed-group audit with two-level bootstrap confidence intervals. "
            "Deltas are LEACE-Race minus Raw Dual."
        ),
        label="tab:seed-paired-audit",
        alignment="lrrrrr",
        headings=[
            "Endpoint",
            "Mean $\\pm$ SD",
            "Seed-group range",
            "Signs (+/--/0)",
            "95\\% CI",
            "Bonferroni CI",
        ],
        rows=rows,
        notes=[
            f"Two-level bootstrap draws: {results['bootstrap']['draws']}.",
            "Bonferroni intervals cover the four paired endpoints.",
        ],
    )


def placebo_table(results: dict[str, Any]) -> str:
    inference = results["inference"]
    primary = inference["primary_null"]
    nulls = inference["null_distributions"][primary]
    tests = inference["one_sided_permutation_tests"]
    rows: list[list[str]] = []
    for metric, label in (
        ("eod_max", r"EOD$_{\max}$"),
        ("dpd", "DPD"),
        ("fpr_gap", "FPR gap"),
    ):
        stats = nulls[metric]
        p = tests[metric]["by_placebo_type"][primary]
        rows.append(
            [
                label,
                f"{number(stats['mean'])} $\\pm$ {number(stats['sd'])}",
                f"[{number(stats['percentile_2_5'])}, {number(stats['percentile_97_5'])}]",
                number(p["p_value"]),
                number(p["holm_adjusted_p_value"]),
            ]
        )
    return table(
        caption=(
            "Primary placebo null distribution and one-sided comparison with true "
            "LEACE. Holm adjustment is across the three placebo types for each metric."
        ),
        label="tab:placebo-v2",
        alignment="lrrrr",
        headings=[
            "Metric",
            "Null mean $\\pm$ SD",
            "Null 2.5--97.5\\%",
            "Raw $p$",
            "Holm-adjusted $p$",
        ],
        rows=rows,
        notes=[
            f"Primary null: {results['primary_null_description']}.",
            "The permutation p-value uses the (k+1)/(N+1) correction.",
        ],
    )


def inlp_table(results: dict[str, Any]) -> str:
    rows = []
    for name in results["method_order"]:
        metrics = results["methods"][name]["metrics"]
        rows.append(
            [
                name,
                number(metrics["auc"]),
                number(metrics["accuracy"]),
                number(metrics["dpd"]),
                number(metrics["eod_max"]),
                number(metrics["linear_probe_macro_auc"]),
                number(metrics["nonlinear_probe_macro_auc"]),
            ]
        )
    return table(
        caption=(
            "Diagnostic and race-decodability results for validation-selected "
            "INLP projections and LEACE."
        ),
        label="tab:inlp-converged",
        alignment="lrrrrrr",
        headings=[
            "Method",
            "AUC",
            "Accuracy",
            "DPD",
            r"EOD$_{\max}$",
            "Linear race AUC",
            "Nonlinear race AUC",
        ],
        rows=rows,
        notes=[
            "INLP stopping and iteration selection use validation probe AUC only.",
            f"Convergence iteration: k={results['selection']['k_at_convergence_or_maximum']}.",
        ],
    )


def baseline_table(results: dict[str, Any]) -> str:
    rows: list[list[str]] = []
    for name in results["table_order"]:
        method = results["methods"][name]
        metrics = method["test_metrics"]
        deltas = method.get("paired_bootstrap_vs_raw", {}).get("metrics", {})
        delta_cells = []
        for key in ("auc", "accuracy", "dpd", "eod_max"):
            ci = deltas.get(key)
            if ci is None:
                delta_cells.append("--")
            else:
                delta_cells.append(
                    f"{number(ci['mean_delta'], signed=True)} "
                    f"[{number(ci['percentile_2_5'], signed=True)},"
                    f"{number(ci['percentile_97_5'], signed=True)}]"
                )
        rows.append(
            [
                name,
                number(metrics["auc"]),
                number(metrics["accuracy"]),
                number(metrics["dpd"]),
                number(metrics["eod_max"]),
                number(metrics["linear_race_probe_macro_auc"]),
                *delta_cells,
            ]
        )
    return table(
        caption=(
            "Five-seed fairness baselines at a fixed threshold. Bootstrap columns "
            "show method-minus-Raw mean delta and 95\\% percentile CI."
        ),
        label="tab:baselines-v2",
        alignment="lrrrrrrrrr",
        headings=[
            "Method",
            "AUC",
            "Accuracy",
            "DPD",
            r"EOD$_{\max}$",
            "Linear race AUC",
            r"$\Delta$ AUC [95\% CI]",
            r"$\Delta$ Acc. [95\% CI]",
            r"$\Delta$ DPD [95\% CI]",
            r"$\Delta$ EOD [95\% CI]",
        ],
        rows=rows,
        notes=[
            "Lambda choices for adversarial and label-conditioned methods were selected on validation.",
            "The lambda=5 label-conditioned failure case is reported separately in the source results.",
        ],
    )


def operating_table(results: dict[str, Any]) -> str:
    rows = []
    for point, section in results["operating_points"].items():
        for endpoint, summary in section["deltas"].items():
            adjusted = results["holm_adjustment"]["tests"][f"{point}:{endpoint}"]
            ci = summary["ci_95_percentile"]
            holm_ci = adjusted["holm_adjusted_ci"]
            rows.append(
                [
                    point.replace("_", r"\ "),
                    endpoint.replace("_", r"\_"),
                    number(summary["point_estimate"], signed=True),
                    f"[{number(ci[0], signed=True)}, {number(ci[1], signed=True)}]",
                    f"[{number(holm_ci[0], signed=True)}, {number(holm_ci[1], signed=True)}]",
                    number(adjusted["holm_adjusted_p_value"]),
                ]
            )
    primary = table(
        caption=(
            "Operating-point sensitivity analysis for canonical Raw Dual and LEACE. "
            "Holm-adjusted intervals use a step-down procedure across 12 endpoint tests."
        ),
        label="tab:operating-points-final",
        alignment="llrrrr",
        headings=[
            "Operating point",
            "Endpoint",
            "Delta",
            "Paired bootstrap 95\\% CI",
            "Holm-adjusted CI",
            "Holm $p$",
        ],
        rows=rows,
        notes=[
            "Cutoffs are selected using validation ensemble predictions only.",
            "Deltas are LEACE minus Raw Dual; bootstrap resamples are paired by test case.",
        ],
    )

    decomposition_rows = []
    for point, section in results["group_gap_decomposition"].items():
        raw, leace = section["raw"], section["leace"]
        delta = section["delta_leace_minus_raw"]
        drivers = section["fpr_gap_change_drivers"]
        shift = drivers["largest_absolute_group_fpr_change_value"]
        decomposition_rows.append(
            [
                point.replace("_", r"\ "),
                number(raw["tpr_gap"]),
                number(leace["tpr_gap"]),
                number(delta["tpr_gap"], signed=True),
                number(raw["fpr_gap"]),
                number(leace["fpr_gap"]),
                number(delta["fpr_gap"], signed=True),
                f"{drivers['raw_max_fpr_group']}/{drivers['raw_min_fpr_group']}",
                f"{drivers['leace_max_fpr_group']}/{drivers['leace_min_fpr_group']}",
                f"{drivers['largest_absolute_group_fpr_change']} ({number(shift, signed=True)})",
            ]
        )
    decomposition = table(
        caption=(
            "TPR and FPR gap decomposition. Group pairs identify maximum/minimum "
            "FPR groups; the final column identifies the largest absolute group FPR shift."
        ),
        label="tab:operating-points-fpr-decomposition",
        alignment="lrrrrrrlll",
        headings=[
            "Operating point",
            "Raw TPR gap",
            "LEACE TPR gap",
            r"$\Delta$ TPR gap",
            "Raw FPR gap",
            "LEACE FPR gap",
            r"$\Delta$ FPR gap",
            "Raw FPR max/min",
            "LEACE FPR max/min",
            "Largest group FPR shift",
        ],
        rows=decomposition_rows,
    )
    return primary + "\n" + decomposition


def modality_tables(results: dict[str, Any]) -> str:
    rows = []
    for key, label in (("image", "Image only"), ("text", "Text only"), ("joint", "Joint")):
        entry = results["modalities"][key]
        rows.append(
            [
                label,
                number(entry["glaucoma"]["auc"]),
                number(entry["glaucoma"]["accuracy_at_0.50"]),
                number(entry["linear_race_probe_macro_auc"]),
            ]
        )
    labels = (
        "Original Raw Dual",
        "Original LEACE-Race",
        "Scrubbed Raw Dual",
        "Scrubbed LEACE-Race",
    )
    for label in labels:
        entry = results["joint_comparison"][label]
        rows.append(
            [
                label,
                number(entry["glaucoma"]["auc"]),
                number(entry["glaucoma"]["accuracy_at_0.50"]),
                number(entry["linear_race_probe_macro_auc"]),
            ]
        )
    modality = table(
        caption=(
            "Modality and text-leakage ablation. Glaucoma accuracy uses a fixed "
            "0.50 cutoff."
        ),
        label="tab:text-modality-ablation",
        alignment="lrrr",
        headings=[
            "Representation",
            "Glaucoma AUC",
            "Glaucoma accuracy",
            "Linear race macro-AUC",
        ],
        rows=rows,
        notes=[
            "Text leakage results are from text_ablation_results.json.",
            "Scrubbed summaries and notes are counted; the original pipeline embeds gpt4_summary.",
        ],
    )
    change_rows = []
    for label, method in (("Raw Dual", "raw"), ("LEACE-Race", "leace")):
        method_result = results["joint_comparison"][
            "Original Raw Dual" if method == "raw" else "Original LEACE-Race"
        ]
        scrubbed_result = results["joint_comparison"][
            "Scrubbed Raw Dual" if method == "raw" else "Scrubbed LEACE-Race"
        ]
        change_rows.append(
            [
                label,
                number(method_result["linear_race_probe_macro_auc"]),
                number(scrubbed_result["linear_race_probe_macro_auc"]),
                number(method_result["nonlinear_race_probe_macro_auc"]),
                number(scrubbed_result["nonlinear_race_probe_macro_auc"]),
                number(method_result["glaucoma"]["eod_max"]),
                number(scrubbed_result["glaucoma"]["eod_max"]),
                number(method_result["glaucoma"]["dpd"]),
                number(scrubbed_result["glaucoma"]["dpd"]),
            ]
        )
    changes = table(
        caption=(
            "Effect of demographic-term scrubbing on joint-feature race "
            "decodability and test disparity."
        ),
        label="tab:text-scrubbing-effect",
        alignment="lrrrrrrrr",
        headings=[
            "Method",
            "Linear AUC pre",
            "Linear AUC post",
            "Nonlinear AUC pre",
            "Nonlinear AUC post",
            "EOD pre",
            "EOD post",
            "DPD pre",
            "DPD post",
        ],
        rows=change_rows,
    )
    return modality + "\n" + changes


def interval_text(values: list[float], signed: bool = True) -> str:
    return f"[{number(values[0], signed=signed)}, {number(values[1], signed=signed)}]"


def abstract_macros(
    canonical: dict[str, Any],
    seed: dict[str, Any],
    placebo: dict[str, Any],
    inlp: dict[str, Any],
    operating: dict[str, Any],
) -> str:
    raw = canonical["ensemble_metrics"]["raw"]
    leace = canonical["ensemble_metrics"]["leace"]
    paired = seed["paired_delta_summary"]
    intervals = seed["bootstrap"]["intervals"]
    primary_type = placebo["inference"]["primary_null"]
    placebo_tests = placebo["inference"]["one_sided_permutation_tests"]
    inlp_leace = inlp["methods"]["LEACE"]["metrics"]
    split_sizes = canonical["split_sanity"]["split_sizes"]
    test_counts = canonical["split_sanity"]["test_race_counts"]
    macros = {
        "AuditTotalN": sum(split_sizes.values()),
        "AuditTrainN": split_sizes["train"],
        "AuditValN": split_sizes["val"],
        "AuditTestN": split_sizes["test"],
        "AuditAsianN": test_counts["Asian"],
        "AuditBlackN": test_counts["Black"],
        "AuditWhiteN": test_counts["White"],
        "AuditSeedGroupCount": len(seed["group_results"]),
        "AuditRawAUC": number(raw["auc"]),
        "AuditLEACEAUC": number(leace["auc"]),
        "AuditRawDPD": number(raw["dpd"]),
        "AuditLEACEDPD": number(leace["dpd"]),
        "AuditRawEOD": number(raw["eod_max"]),
        "AuditLEACEEOD": number(leace["eod_max"]),
        "AuditLinearProbeAUC": number(placebo["true_leace"]["race_probe_macro_auc"]),
        "AuditNonlinearProbeAUC": number(inlp_leace["nonlinear_probe_macro_auc"]),
        "AuditDeltaAUC": number(paired["d_auc"]["mean"], signed=True),
        "AuditDeltaAUCCI": interval_text(intervals["d_auc"]["ci_95_percentile"]),
        "AuditDeltaAUCAdjustedCI": interval_text(
            intervals["d_auc"]["bonferroni_adjusted_ci"]
        ),
        "AuditDeltaDPD": number(paired["d_dpd"]["mean"], signed=True),
        "AuditDeltaDPDCI": interval_text(intervals["d_dpd"]["ci_95_percentile"]),
        "AuditDeltaDPDAdjustedCI": interval_text(
            intervals["d_dpd"]["bonferroni_adjusted_ci"]
        ),
        "AuditDeltaEOD": number(paired["d_eod_max"]["mean"], signed=True),
        "AuditDeltaEODAdjustedCI": interval_text(
            intervals["d_eod_max"]["bonferroni_adjusted_ci"]
        ),
        "AuditPlaceboEODp": number(
            placebo_tests["eod_max"]["by_placebo_type"][primary_type][
                "holm_adjusted_p_value"
            ]
        ),
        "AuditPlaceboDPDp": number(
            placebo_tests["dpd"]["by_placebo_type"][primary_type][
                "holm_adjusted_p_value"
            ]
        ),
        "AuditFixedOperatingPointDPDp": number(
            operating["holm_adjustment"]["tests"]["fixed_0.50:d_dpd"][
                "holm_adjusted_p_value"
            ]
        ),
    }
    lines = [
        "% Generated by src/make_tables.py; do not edit values by hand.",
        *[rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items()],
        "",
    ]
    return "\n".join(lines)


def numbers_summary(
    canonical: dict[str, Any],
    seed: dict[str, Any],
    placebo: dict[str, Any],
    inlp: dict[str, Any],
) -> str:
    lines = [
        "ABSTRACT NUMBERS - GENERATED FROM SAVED JSON RESULTS",
        "Canonical model metrics (placebo-audit run; fixed recorded threshold)",
        "Method       AUC      Accuracy  DPD      EOD_max  Worst-group AUC",
    ]
    for key, label in (("raw", "Raw Dual"), ("leace", "LEACE-Race")):
        m = canonical["ensemble_metrics"][key]
        lines.append(
            f"{label:<12} {m['auc']:.4f}   {m['acc']:.4f}    "
            f"{m['dpd']:.4f}   {m['eod_max']:.4f}   {m['w_auc']:.4f}"
        )
    split = canonical["split_sanity"]["split_sizes"]
    counts = canonical["split_sanity"]["test_race_counts"]
    lines.extend(
        [
            f"Split sizes: train={split['train']}, validation={split['val']}, "
            f"test={split['test']}; test race counts Asian/Black/White="
            f"{counts['Asian']}/{counts['Black']}/{counts['White']}.",
            "",
            (
                f"Paired seed groups: {len(seed['group_results'])}; "
                f"seeds per group: {len(seed['group_definitions'][0]['seeds'])}; "
                f"two-level bootstrap draws: {seed['bootstrap']['draws']}."
            ),
            "Paired seed-group deltas (LEACE minus Raw Dual)",
        ]
    )
    for key, label in (
        ("d_auc", "dAUC"),
        ("d_eod_max", "dEOD_max"),
        ("d_dpd", "dDPD"),
    ):
        summary = seed["paired_delta_summary"][key]
        interval = seed["bootstrap"]["intervals"][key]
        lines.append(
            f"{label}: mean={summary['mean']:+.4f}, SD={summary['sd']:.4f}; "
            f"95% CI={interval_text(interval['ci_95_percentile'])}; "
            f"Bonferroni CI={interval_text(interval['bonferroni_adjusted_ci'])}."
        )

    lines.extend(["", "Race probe macro-AUCs (separate experiments; do not pool)"])
    linear = placebo["true_leace"]["race_probe_macro_auc"]
    nonlinear = inlp["methods"]["LEACE"]["metrics"]["nonlinear_probe_macro_auc"]
    lines.append(f"Placebo audit, true LEACE linear probe: {linear:.4f}")
    lines.append(f"Converged-INLP audit, LEACE nonlinear probe: {nonlinear:.4f}")

    tests = placebo["inference"]["one_sided_permutation_tests"]
    primary = placebo["inference"]["primary_null"]
    lines.extend(
        [
            "",
            f"Placebo p-values (primary null: {primary}; raw / Holm over placebo types)",
        ]
    )
    for metric in ("eod_max", "dpd", "fpr_gap"):
        vals = tests[metric]["by_placebo_type"][primary]
        lines.append(
            f"{metric:<10} p={vals['p_value']:.4f}, Holm-p={vals['holm_adjusted_p_value']:.4f}"
        )

    lines.extend(
        [
            "",
            "Primary placebo null distributions (mean +/- SD; 2.5th-97.5th percentiles)",
        ]
    )
    nulls = placebo["inference"]["null_distributions"]
    for metric in ("eod_max", "dpd", "fpr_gap"):
        stat = nulls[primary][metric]
        lines.append(
            f"{metric:<10} {stat['mean']:.4f} +/- {stat['sd']:.4f} "
            f"[{stat['percentile_2_5']:.4f}, {stat['percentile_97_5']:.4f}] "
            f"(N={stat['n']})"
        )
    lines.extend(
        [
            "",
            "Sources: canonical_results.json; seed_paired_results.json;",
            "placebo_v2_results.json; inlp_converged_results.json.",
            "Values are kept run-specific; results from distinct protocols are not combined.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    tables_dir = root / "tables"

    canonical = load_json(root / "canonical_results.json")
    seed = load_json(root / "seed_paired_results.json")
    placebo = load_json(root / "placebo_v2_results.json")
    inlp = load_json(root / "inlp_converged_results.json")
    baselines = load_json(root / "baselines_v2_results.json")
    operating = load_json(root / "operating_points_final.json")
    text_ablation = load_json(root / "text_ablation_results.json")

    generated = {
        "canonical.tex": canonical_table(canonical),
        "seed_paired.tex": seed_table(seed),
        "placebo_v2.tex": placebo_table(placebo),
        "inlp.tex": inlp_table(inlp),
        "baselines.tex": baseline_table(baselines),
        "opsens.tex": operating_table(operating),
        "modality.tex": modality_tables(text_ablation),
        "abstract_numbers.tex": abstract_macros(
            canonical, seed, placebo, inlp, operating
        ),
    }
    tables_dir.mkdir(parents=True, exist_ok=True)
    for filename, contents in generated.items():
        (tables_dir / filename).write_text(contents, encoding="utf-8")

    summary = numbers_summary(canonical, seed, placebo, inlp)
    summary_path = root / "numbers_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"\nGenerated {len(generated) - 1} LaTeX table files and abstract macros in {tables_dir}.")
    print(f"Saved numbers summary to {summary_path}.")


if __name__ == "__main__":
    main()
