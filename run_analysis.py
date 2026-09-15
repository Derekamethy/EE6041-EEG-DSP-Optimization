from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.demo_data import generate_demo_eeg
from src.dsp_pipeline import (
    FS_ORIGINAL,
    FS_TARGET,
    benchmark_resamplers,
    comparison_metrics,
    design_resample_fir,
    epoch_psd,
    filter_response_metrics,
    iwmf_features,
    load_eeg_xlsx,
    normalize_psd,
    resample_direct_reference,
    resample_polyphase,
    segment_epochs,
    spectral_preservation_metrics,
    theoretical_complexity,
)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def build_metrics(data):
    taps = design_resample_fir()
    reference = resample_direct_reference(data, taps)
    polyphase = resample_polyphase(data, taps)

    overlap = min(len(reference), len(polyphase))
    numerical_equivalence = comparison_metrics(
        reference[:overlap],
        polyphase[:overlap],
    )

    raw_epochs, raw_times = segment_epochs(data, FS_ORIGINAL)
    poly_epochs, poly_times = segment_epochs(polyphase, FS_TARGET)
    raw_iwmf = iwmf_features(raw_epochs, FS_ORIGINAL)
    poly_iwmf = iwmf_features(poly_epochs, FS_TARGET)

    spectral_validation = spectral_preservation_metrics(
        raw_epochs,
        FS_ORIGINAL,
        poly_epochs,
        FS_TARGET,
    )
    spectral_validation["iwmf"] = comparison_metrics(raw_iwmf, poly_iwmf)

    raw_freqs, raw_psd = epoch_psd(raw_epochs, FS_ORIGINAL)
    poly_freqs, poly_psd = epoch_psd(poly_epochs, FS_TARGET)
    if not (
        len(raw_freqs) == len(poly_freqs)
        and np.allclose(raw_freqs, poly_freqs, atol=1e-12)
    ):
        raise RuntimeError("PSD frequency grids are not aligned for plotting")

    metrics = {
        "input": {
            "samples": int(len(data)),
            "sampling_rate_hz": FS_ORIGINAL,
            "duration_seconds": float(len(data) / FS_ORIGINAL),
        },
        "resampling": {
            "output_sampling_rate_hz": FS_TARGET,
            "output_samples": int(len(polyphase)),
            "expected_output_samples": int(
                round(len(data) * FS_TARGET / FS_ORIGINAL)
            ),
            "reference_overlap_samples": int(overlap),
        },
        "anti_alias_filter": filter_response_metrics(taps),
        "frequency_resolution_hz": {
            "500_hz_8s_epoch": float(FS_ORIGINAL / raw_epochs.shape[1]),
            "32_hz_8s_epoch": float(FS_TARGET / poly_epochs.shape[1]),
        },
        "numerical_equivalence": numerical_equivalence,
        "spectral_validation_target_band": spectral_validation,
        "theoretical_complexity": theoretical_complexity(len(data)),
        "runtime_benchmark": benchmark_resamplers(data, taps),
    }

    series = {
        "reference": reference,
        "polyphase": polyphase,
        "raw_times": raw_times,
        "poly_times": poly_times,
        "raw_iwmf": raw_iwmf,
        "poly_iwmf": poly_iwmf,
        "psd_freqs": raw_freqs,
        "raw_mean_normalized_psd": normalize_psd(raw_psd).mean(axis=0),
        "poly_mean_normalized_psd": normalize_psd(poly_psd).mean(axis=0),
    }
    return metrics, series


def save_figures(series, prefix):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    n = min(
        len(series["reference"]),
        len(series["polyphase"]),
        int(4 * FS_TARGET),
    )
    t = np.arange(n) / FS_TARGET

    plt.figure(figsize=(10, 4))
    plt.plot(
        t,
        series["reference"][:n],
        label="Direct-form FIR reference",
        linewidth=1.4,
    )
    plt.plot(
        t,
        series["polyphase"][:n],
        "--",
        label="Optimised polyphase FIR",
        linewidth=1.1,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Direct-Form and Polyphase FIR Output Equivalence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / f"{prefix}_resampling_equivalence.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(
        series["raw_times"],
        series["raw_iwmf"],
        marker="o",
        label="500 Hz reference",
    )
    plt.plot(
        series["poly_times"],
        series["poly_iwmf"],
        "--o",
        label="32 Hz polyphase",
    )
    plt.xlabel("Epoch centre time (s)")
    plt.ylabel("IWMF (Hz)")
    plt.title("IWMF Preservation Across 500 Hz to 32 Hz Resampling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{prefix}_iwmf_preservation.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(
        series["psd_freqs"],
        series["raw_mean_normalized_psd"],
        label="500 Hz reference",
        linewidth=1.5,
    )
    plt.plot(
        series["psd_freqs"],
        series["poly_mean_normalized_psd"],
        "--",
        label="32 Hz polyphase",
        linewidth=1.2,
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Mean normalized PSD")
    plt.title("0.5-12.5 Hz PSD Preservation After Resampling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{prefix}_psd_preservation.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run the multirate EEG DSP pipeline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional single-column XLSX EEG file. Uses the synthetic demo if omitted.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional label for generated result and figure filenames.",
    )
    args = parser.parse_args()

    if args.input is None:
        data = generate_demo_eeg(fs=FS_ORIGINAL, seconds=60.0)
        label = args.label or "demo"
        source = "deterministic synthetic EEG-like demo"
    else:
        data = load_eeg_xlsx(args.input)
        label = args.label or args.input.stem
        source = args.input.name

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics, series = build_metrics(data)
    metrics["source"] = {"label": label, "description": source}
    save_figures(series, label)

    output = RESULTS_DIR / f"{label}_metrics.json"
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics: {output}")
    print(f"Saved figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()