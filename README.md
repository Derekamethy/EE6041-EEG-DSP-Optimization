# EEG Multirate DSP Pipeline with Polyphase FIR Optimisation

A complete EEG signal-processing pipeline for efficient **500 Hz -> 32 Hz** resampling, anti-alias filtering, epoch-based spectral analysis, and quantitative information-preservation validation.

The system uses rational multirate processing with `L = 8` and `M = 125`. A specification-driven Kaiser FIR protects the 0.5-12.5 Hz analysis band before decimation, while a polyphase implementation avoids most of the computation required by a direct upsample-filter-decimate realization.

## Key results

Validated on a 60 s, 30,000-sample EEG recording:

| Metric | Result |
| --- | ---: |
| Input / output sampling rate | 500 Hz -> 32 Hz |
| Output samples | 1,920 |
| FFT resolution for 8 s epochs | 0.125 Hz |
| Direct-form vs polyphase correlation | 1.000000 |
| Direct-form vs polyphase MSE | 7.77e-27 |
| Median PSD-shape correlation | 0.999999996 |
| Median normalized-PSD relative RMSE | 0.0091% |
| Median band-power error, 0.5-4 Hz | 0.0277% |
| Median band-power error, 4-8 Hz | 0.0263% |
| Median band-power error, 8-12.5 Hz | 0.0318% |
| Spectral-entropy correlation | 0.999999998 |
| SEF95 RMSE | 0.000 Hz |
| IWMF RMSE / correlation | 0.0412 Hz / 0.9911 |
| Theoretical FIR-operation reduction | ~1000x |
| Measured median runtime speed-up | ~119x |

The runtime result is machine-dependent. The operation-count comparison follows from the two FIR implementations and is reported separately from measured wall-clock timing.

## Signal-processing pipeline

```text
500 Hz EEG
   |
   |  Rational conversion L = 8, M = 125
   v
Specification-driven Kaiser anti-alias FIR
   |
   |  Passband edge: 12.5 Hz
   |  Stopband edge: 16 Hz
   |  Stopband target: 60 dB
   v
32 Hz EEG
   |
8 s epochs, 50% overlap
   |
Hann-window PSD / spectral features
   v
PSD shape + band power + entropy + SEF95 + IWMF validation
```

The 8 s analysis window produces the same frequency-bin spacing on both branches:

`500 / 4000 = 32 / 256 = 0.125 Hz`.

## Anti-alias filter design

The 32 Hz target rate has a 16 Hz Nyquist frequency. The anti-alias filter is therefore designed from explicit spectral constraints rather than an arbitrary tap count.

The Kaiser design uses a 3.5 Hz transition band at the 4 kHz interpolation rate and yields **4,145 taps** with beta = 5.653. Measured frequency-response checks give:

| Filter check | Achieved |
| --- | ---: |
| Gain at 12.5 Hz | -0.0095 dB |
| Worst passband deviation | 0.0090 dB |
| Gain at 16 Hz | -60.32 dB |
| Minimum stopband attenuation | 59.89 dB |

This keeps the complete 0.5-12.5 Hz target analysis band essentially flat while strongly suppressing content at and above the new Nyquist limit.

## Polyphase optimisation

A direct-form rational resampler explicitly creates the 8x zero-stuffed sequence and evaluates a long FIR at the 4 kHz intermediate rate. The polyphase form reorganises the same coefficients so that only contributions required for retained output samples are evaluated.

For the 30,000-sample validation recording:

```text
Direct form:  30,000 x 8 x 4,145    = 994,800,000 MACs
Polyphase:    30,000 x 4,145 / 125  =     994,800 MACs (approx.)
```

The estimated core FIR workload is therefore reduced by approximately **1000x**. On the validation machine, the measured median runtime improved by approximately **119x**.

The efficient and direct-form implementations remain numerically equivalent over their common valid output region:

- MSE: `7.77e-27`
- RMSE: `8.81e-14`
- Pearson correlation: `1.000000`

![Direct-form and polyphase equivalence](figures/resampling_equivalence.png)

## Target-band spectral preservation

Information preservation is evaluated across the full **0.5-12.5 Hz** target band, not only with a single summary feature. Each 8 s epoch is analysed with a Hann-window periodogram on an aligned 0.125 Hz frequency grid.

### PSD shape

The normalized PSD curves from the 500 Hz reference and 32 Hz output are nearly indistinguishable:

- Median PSD correlation: **0.999999996**
- Median normalized-PSD relative RMSE: **0.0091%**

![PSD preservation](figures/psd_preservation.png)

### Band-power preservation

Integrated spectral power is compared in three sub-bands:

| Band | Median relative error |
| --- | ---: |
| 0.5-4 Hz | 0.0277% |
| 4-8 Hz | 0.0263% |
| 8-12.5 Hz | 0.0318% |

### Spectral summary features

The validation also checks complementary descriptors of spectral structure:

| Feature | Result |
| --- | ---: |
| Spectral entropy | r = 0.999999998 |
| SEF95 | RMSE = 0.000 Hz, r = 1.000 |
| IWMF | RMSE = 0.0412 Hz, r = 0.9911 |

IWMF describes the power-weighted spectral centre, spectral entropy measures how concentrated or distributed the band-limited power is, and SEF95 tracks the frequency below which 95% of the target-band power is contained.

![IWMF preservation](figures/iwmf_preservation.png)

## Validation framework

The project separates four questions that are often conflated in downsampling validation:

1. **Implementation equivalence** 鈥?does the polyphase implementation match the direct-form FIR result?
2. **Full-spectrum preservation** 鈥?is the complete normalized PSD shape retained in the target band?
3. **Band-energy preservation** 鈥?are integrated powers retained across low, mid and high sub-bands?
4. **Feature preservation** 鈥?are complementary spectral descriptors such as entropy, SEF95 and IWMF retained?

This structure makes the validation stronger than relying on a single frequency-domain statistic.

## Repository structure

```text
.
|-- src/
|   |-- dsp_pipeline.py        # multirate DSP and spectral validation functions
|   `-- demo_data.py           # deterministic EEG-like public demo signal
|-- tests/
|   `-- test_pipeline.py       # numerical, filter and spectral regression tests
|-- results/
|   `-- benchmark_results.json # validated EEG benchmark summary
|-- figures/
|   |-- resampling_equivalence.png
|   |-- psd_preservation.png
|   `-- iwmf_preservation.png
|-- run_analysis.py
|-- requirements.txt
`-- .gitignore
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

Running `run_analysis.py` without an input file uses a deterministic EEG-like signal so the complete pipeline can be executed without external data.

To analyse a single-column XLSX recording:

```bash
python run_analysis.py --input path/to/recording.xlsx --label recording_name
```

The validation recording itself is not redistributed. The repository contains the derived benchmark summary and figures only.

## Implementation highlights

- Python, NumPy and SciPy signal-processing pipeline
- 500 Hz -> 32 Hz rational multirate conversion (`8/125`)
- Specification-driven 4,145-tap Kaiser anti-alias FIR
- Polyphase FIR resampling with `scipy.signal.resample_poly`
- 8 s epochs with 50% overlap and 0.125 Hz spectral resolution
- Hann-window periodogram PSD analysis
- Full-band PSD correlation and normalized spectral error
- Three-band integrated-power validation
- Spectral entropy, SEF95 and IWMF feature validation
- Automated tests for filter specifications, output length, numerical equivalence, PSD alignment and spectral metrics

## Scope

This project evaluates information preservation within the **0.5-12.5 Hz target band** after 500 Hz -> 32 Hz resampling. Frequencies above the 16 Hz Nyquist limit of the target rate are intentionally removed by anti-alias filtering and are not claimed to be preserved.

The repository demonstrates multirate DSP design, computational optimisation, spectral analysis and signal-level validation. It does not make clinical or downstream diagnostic-performance claims.

