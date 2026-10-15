[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)

# Metastability on Firm Ground: A Generalized Stochastic Model for Metastability-based TRNGs

## 📝 About

This repository contains the data, analysis code, and figures accompanying **Metastability on Firm Ground: A Generalized Stochastic Model for Metastability-based TRNGs**.

The paper connects circuit behavior, output-sequence statistics, and entropy estimation through the effective metastable state. The model decomposes this state into a static offset, cycle-local white noise, and correlated flicker noise, and evaluates average conditional min-entropy over finite blocks.

The experiments examine this model through transistor-level simulation, FPGA measurements, and a 28 nm CMOS prototype. FPGA captures show reproducible power-law autocorrelation tails after bias calibration. ASIC measurements span 24 operating corners across three temperatures. At 27 °C, the model estimates 0.237–0.412 bits per sample for 256-bit blocks, accounting for temporal correlations that bias-only i.i.d. estimates miss.

## 🖥 Code execution

Use Python 3.12 or newer. The validated environments are Python 3.14.3 on Windows and Python 3.12.15 on Ubuntu 22.04, with the versions in [requirements-lock.txt](requirements-lock.txt).

### Run locally

From the repository root, create a virtual environment:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS, then run:

```sh
python -m pip install -r requirements-lock.txt
python reproduce.py
python verify.py
```

The default workflow runs locally using the included datasets after dependency installation. It generates nine quantitative figures, two tables, and the Section 5 result summary. Monte Carlo figures and table estimates are rendered from the paper's reference CSVs; the script overview below links to the corresponding computation workflows.

### Run selected experiments

Pass one or more result names to the runner:

```sh
python reproduce.py fig10 fig13
python reproduce.py table01 table02
```

Table 1 automatically runs the Figure 8 analysis first. To extract ASIC parameters from the included captures and plot Figures 12 and 14, use:

```sh
python reproduce.py --asic-from-raw fig12 fig14
```

Plots are saved in each experiment's `figure/` directory. Analysis outputs are saved in `data/derived/`, and optional recomputation workflows use the output paths documented in their local READMEs.

## 📁 Data

The repository includes Spectre waveform exports, 30 packed FPGA captures, 250 ASIC capture candidates, and ten synchronized pairs of ASIC entropy-source records. Reference statistics and parameter tables accompany the measurements.

```text
├── Spectre/                 Transistor-level waveform analysis
│   ├── fig04/               Output probability and effective offset
│   ├── fig05/               Metastability resolution-time analysis
│   ├── fig08/               Flicker-noise sequence statistics
│   └── table01/             Bias and SP 800-90B entropy estimates
├── Model/                   Finite-block entropy calculations
│   ├── fig06/               Sampling-frequency dependence
│   └── fig07/               Offset, flicker-noise, and block-length dependence
├── FPGA/fig10/              Calibrated FPGA captures and autocorrelation tails
├── ASIC/
│   ├── raw_reproduction/    Capture selection and parameter extraction
│   ├── fig12/               Latent autocorrelation across three temperatures
│   ├── fig13/               Cross-correlation between two entropy sources
│   ├── fig14/               Split-sample model validation
│   ├── table02/             Measured and model-based entropy estimates
│   └── sensitivity/         Kernel, block-length, and trajectory-tail analyses
└── nist90b/                 SP 800-90B computation and verification
```

<details>
<summary>Data formats and acquisition settings</summary>

Figure 10 uses packed bytes decoded least-significant bit first. Each of the 30 captures contains 1,000,000 bits, with ten captures at each of 10, 25, and 50 MHz. Figure 13 uses one byte per bit and contains ten synchronized pairs of 1,000,000-symbol records.

The ASIC capture collection contains 240 records for split validation and ten additional acquisition candidates used in minimum-bias selection. The [capture index](ASIC/raw_reproduction/captures.csv) records filenames, hashes, sample counts, acquisition settings, and repeat indices.

ASIC time parameters use oscilloscope sample ticks at 80 GSa/s (12.5 ps per tick). Frequency settings F1–F4 correspond to 1–4 GHz. See the [raw-data guide](ASIC/raw_reproduction/README.md) for selection rules, unit conversions, and parameter extraction.

</details>

## 📊 Scripts overview

### 🔬 Circuit-level analysis

The Spectre workflows connect waveform measurements to output probability, resolution time, and temporal correlation. They analyze the included simulator exports using Python.

| Paper result | Analysis guide | Workflow |
|---|---|---|
| Figure 4 | [Output probability](Spectre/fig04/README.md) | Waveforms → probabilities → Gaussian-CDF fit |
| Figure 5 | [Resolution time](Spectre/fig05/README.md) | Waveforms → threshold crossings → fitted distribution |
| Figure 8 | [Flicker-noise comparison](Spectre/fig08/README.md) | Waveforms → bits → autocorrelation and block statistics |
| Table 1 | [Entropy comparison](Spectre/table01/README.md) | Figure 8 statistics + reference NIST estimates → table |

### 💻 Stochastic model

[Model/README.md](Model/README.md) describes Figures 6 and 7, which explore how sampling frequency, static offset, flicker-noise strength, and block length shape conditional min-entropy. The default commands plot the reference estimates. The guide also provides Monte Carlo commands, parameters, and seeds for new calculations.

### 📈 FPGA measurements

The [Figure 10 workflow](FPGA/fig10/README.md) computes autocorrelation directly from packed captures, fits the power-law tails, and evaluates bootstrap intervals and block-variance inflation. These statistics characterize temporal correlation after bias calibration.

### 🌡️ ASIC measurements and validation

| Paper result | Analysis guide | Workflow |
|---|---|---|
| Figure 12 | [Latent autocorrelation](ASIC/fig12/README.md) | Captures → threshold inversion → power-law fits |
| Figure 13 | [Dual-source measurements](ASIC/fig13/README.md) | Synchronized bitstreams → block means → cross-correlation |
| Figure 14 | [Split validation](ASIC/fig14/README.md) | Independent capture splits → parameter fits → held-out predictions |
| Table 2 | [Conditional min-entropy](ASIC/table02/README.md) | Extracted parameters → model estimates and NIST comparison |
| Section 5 | [Sensitivity analysis](ASIC/sensitivity/README.md) | Alternative kernels, block lengths, and trajectory-tail quantiles |

Figures 12 and 14 and Table 2 use reference CSVs in the default workflow. The [raw-data guide](ASIC/raw_reproduction/README.md) supplies the capture-based parameter extraction commands, and the local guides describe subsequent entropy calculations and validation.

### 🔎 SP 800-90B assessment

[nist90b/README.md](nist90b/README.md) provides build instructions for the pinned NIST implementation and commands to assess the included Spectre and ASIC records. Each run saves entropy estimates, test decisions, input hashes, and execution logs for comparison with Tables 1 and 2.
