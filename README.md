# EEG Multirate DSP Pipeline with Polyphase FIR Optimisation

Efficient **500 Hz → 32 Hz** EEG resampling with specification-driven anti-alias filtering, polyphase optimisation, and quantitative spectral-preservation validation.

[![CI](https://github.com/Derekamethy/eeg-multirate-dsp/actions/workflows/ci.yml/badge.svg)](https://github.com/Derekamethy/eeg-multirate-dsp/actions/workflows/ci.yml)

**UCC EE6041 Advanced Digital Signal Processing · Yangdeyi Yang**

The pipeline uses rational multirate processing with `L = 8` and `M = 125`. A Kaiser FIR protects the **0.5–12.5 Hz** analysis band before decimation, while the polyphase implementation avoids most of the computation required by a direct upsample-filter-decimate realization.

## At a glance

Validated on a 60 s, 30,000-sample EEG recording:

| Metric | Result |
| --- | ---: |
| Input / output sampling rate | 500 Hz → 32 Hz |
| Output samples | 1,920 |
| FFT resolution for 8 s epochs | 0.125 Hz |
| Direct-form vs polyphase correlation | 1.000000 |
| Direct-form vs polyphase MSE | 7.77e-27 |
| Median PSD-shape correlation | 0.999999996 |
| Median normalized-PSD relative RMSE | 0.0091% |
| Median band-power error across 0.5–12.5 Hz sub-bands | 0.0263–0.0318% |
| Spectral-entropy correlation | 0.999999998 |
| IWMF RMSE / correlation | 0.0412 Hz / 0.9911 |
| Theoretical FIR-operation reduction | ~1000× |
| Measured median runtime speed-up | ~119× |

The runtime figure is machine-dependent. The ~1000× figure is a theoretical FIR-operation comparison and is reported separately from measured wall-clock timing.

![Direct-form and polyphase output equivalence](figures/resampling_equivalence.png)

## Why this problem matters

Reducing EEG from 500 Hz to 32 Hz can cut storage and downstream computation dramatically, but only if the information needed for later spectral analysis is preserved. A downsampling pipeline therefore has to answer two engineering questions at the same time:

- **Is aliasing controlled by a filter derived from explicit spectral requirements?**
- **Can the same resampling operation be implemented much more efficiently without changing the retained signal?**

This project treats resampling as a complete DSP design-and-validation problem rather than as a single library call.

## My contribution

- Designed the **500 Hz → 32 Hz** rational conversion with `L = 8`, `M = 125`.
- Derived a specification-driven Kaiser anti-alias FIR from the 12.5 Hz passband edge, 16 Hz stopband edge, and 60 dB attenuation target.
- Implemented a direct upsample-filter-decimate reference and an efficient polyphase realization using the same FIR coefficients.
- Verified numerical equivalence between the two implementations before using the polyphase form for the efficient pipeline.
- Built an epoch-level spectral validation framework covering PSD shape, sub-band power, spectral entropy, SEF95, and IWMF.
- Quantified theoretical FIR workload and measured runtime speed-up separately to avoid conflating operation counts with machine-dependent timing.
- Added deterministic synthetic demo data and automated regression tests so the pipeline can be executed without redistributing the validation recording.

## Signal-processing pipeline

```text
500 Hz EEG
   ↓
Rational conversion: L = 8, M = 125
   ↓
Specification-driven Kaiser anti-alias FIR
   ↓
32 Hz EEG
   ↓
8 s epochs, 50% overlap
   ↓
Hann-window PSD + spectral features
   ↓
PSD shape + band power + entropy + SEF95 + IWMF validation
```

The 8 s analysis window gives identical frequency-bin spacing on both branches:

`500 / 4000 = 32 / 256 = 0.125 Hz`.

## Engineering decisions

| Decision | Rationale | Boundary / trade-off |
| --- | --- | --- |
| Rational factor `8/125` | Converts 500 Hz exactly to 32 Hz | Requires a high-rate interpolation filter before decimation |
| 12.5 Hz protected band | Preserves the full target analysis range with margin below the 16 Hz output Nyquist limit | Content above 16 Hz is intentionally discarded |
| Specification-driven Kaiser FIR | Converts passband/stopband requirements into a reproducible design | Produces a long 4,145-tap reference filter |
| Direct-form implementation | Provides a transparent numerical reference | Computationally expensive and not the intended efficient path |
| Polyphase implementation | Evaluates only contributions needed for retained output samples | Requires careful equivalence validation |
| 8 s, 50%-overlap epochs | Gives 0.125 Hz spectral resolution at both sampling rates | Longer windows reduce temporal resolution |
| Multiple spectral metrics | Tests shape, energy, and summary-feature preservation separately | More informative than a single scalar metric, but not a downstream clinical validation |

## Anti-alias filter design

The 32 Hz target rate has a 16 Hz Nyquist frequency. The filter is therefore derived from explicit constraints rather than an arbitrary tap count:

- passband edge: **12.5 Hz**
- stopband edge: **16 Hz**
- stopband target: **60 dB**
- interpolation rate: **4 kHz**
- resulting FIR length: **4,145 taps**
- Kaiser beta: **5.653**

Measured response checks:

| Filter check | Achieved |
| --- | ---: |
| Gain at 12.5 Hz | -0.0095 dB |
| Worst passband deviation | 0.0090 dB |
| Gain at 16 Hz | -60.32 dB |
| Minimum stopband attenuation | 59.89 dB |

## Polyphase optimisation

The direct reference explicitly creates the 8× zero-stuffed sequence and evaluates a 4,145-tap FIR at the 4 kHz intermediate rate. The polyphase form reorganises the same coefficients so only contributions required for retained output samples are evaluated.

For the 30,000-sample validation recording:

```text
Direct form:  30,000 × 8 × 4,145    = 994,800,000 MACs
Polyphase:    30,000 × 4,145 / 125  ≈     994,800 MACs
```

The estimated core FIR workload is therefore reduced by approximately **1000×**. On the validation machine, the median wall-clock runtime improved by approximately **119×**.

The efficient and direct implementations remain numerically equivalent over their common valid output region:

- MSE: `7.77e-27`
- RMSE: `8.81e-14`
- Pearson correlation: `1.000000`

## Spectral preservation

Information preservation is evaluated across the full **0.5–12.5 Hz** target band, not with a single summary statistic.

### PSD shape

- Median PSD correlation: **0.999999996**
- Median normalized-PSD relative RMSE: **0.0091%**

![PSD preservation](figures/psd_preservation.png)

### Band power

| Band | Median relative error |
| --- | ---: |
| 0.5–4 Hz | 0.0277% |
| 4–8 Hz | 0.0263% |
| 8–12.5 Hz | 0.0318% |

### Spectral summary features

| Feature | Result |
| --- | ---: |
| Spectral entropy | r = 0.999999998 |
| SEF95 | RMSE = 0.000 Hz, r = 1.000 |
| IWMF | RMSE = 0.0412 Hz, r = 0.9911 |

![IWMF preservation](figures/iwmf_preservation.png)

## Validation framework

The validation separates four distinct questions:

1. **Implementation equivalence** — does the polyphase implementation match the direct-form FIR reference?
2. **Full-spectrum preservation** — is the normalized PSD shape retained throughout the target band?
3. **Band-energy preservation** — are integrated powers retained across low, mid, and high sub-bands?
4. **Feature preservation** — are complementary spectral descriptors such as entropy, SEF95, and IWMF retained?

Keeping these questions separate makes the evidence stronger than relying on one frequency-domain statistic.

## Repository structure

```text
.
├── src/
│   ├── dsp_pipeline.py        # multirate DSP and spectral validation functions
│   └── demo_data.py           # deterministic EEG-like demo signal
├── tests/
│   └── test_pipeline.py       # numerical, filter, and spectral regression tests
├── results/
│   └── benchmark_results.json # validated benchmark summary
├── figures/                   # equivalence and spectral-preservation figures
├── run_analysis.py            # CLI entry point
├── requirements.txt
└── .github/workflows/ci.yml   # automated test workflow
```

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_analysis.py
python -m unittest discover -s tests -v
```

Running `run_analysis.py` without an input file uses a deterministic EEG-like signal so the full pipeline can be executed without external data.

To analyse a single-column XLSX recording:

```bash
python run_analysis.py --input path/to/recording.xlsx --label recording_name
```

The validation recording itself is not redistributed. The repository includes the derived benchmark summary and figures needed to inspect the reported results.

## Reproducibility

`tests/test_pipeline.py` checks output length, FIR specifications, direct/polyphase numerical equivalence, epoch frequency resolution, PSD-grid alignment, normalized PSD behaviour, spectral features, and the theoretical workload reduction.

The CI workflow runs these regression tests on every push and pull request. Machine-dependent runtime numbers are intentionally kept separate from deterministic numerical and spectral checks.

## Scope and limitations

This project evaluates information preservation within the **0.5–12.5 Hz target band** after 500 Hz → 32 Hz resampling. Frequencies above the 16 Hz Nyquist limit of the target rate are intentionally removed by anti-alias filtering and are not claimed to be preserved.

The repository demonstrates multirate DSP design, computational optimisation, spectral analysis, and signal-level validation. It does **not** make clinical, diagnostic, seizure-detection, or downstream model-performance claims.

## License

Source code is provided under the MIT License. Third-party data and externally owned materials retain their original rights and terms.
