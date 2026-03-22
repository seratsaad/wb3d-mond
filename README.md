# wb3d-mond

**High-Precision Differential Radial Velocities of C3PO Wide Binaries: A Test of Modified Newtonian Dynamics (MOND)**

Serat M. Saad & Yuan-Sen Ting (2026)

[arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) | [ADS](https://ui.adsabs.harvard.edu/)

---

## Overview

This repository contains the data and analysis code for Saad & Ting (2026), in which we test MOND using high-precision differential radial velocities of wide-binary stars from the C3PO survey.

We measure differential RVs with precisions of ~8–15 m/s per binary pair using high-resolution echelle spectroscopy, a ~24× improvement over *Gaia* DR3. We construct a hierarchical Bayesian model to jointly infer orbital elements for 57 gravitationally bound wide-binary systems and the global MOND acceleration scale *a*₀.

### Key Results

- **b = 1 (simple interpolating function):** canonical *a*₀ excluded at 3.1σ
- **b = 2 (standard interpolating function):** canonical *a*₀ excluded at 1.9σ
- **Gravity boost factor:** γ = 1.08 ± 0.20, consistent with Newtonian gravity

---

## Repository Structure

```
wb3d-mond/
├── README.md
├── mond_analysis.py                 # Main analysis script (all three analyses)
├── data/
│   └── c3po_wide_binaries.csv       # Differential RV measurements + Gaia astrometry for 85 C3PO pairs
└── notebooks/
    └── mond_hierarchical_model.ipynb # Full development notebook (exploratory)
```

## Data

`data/c3po_wide_binaries.csv` contains the full dataset of 85 C3PO wide-binary pairs with:

- **Differential RV measurements** (`rv_diff`, `rv_sigma`) from our pixel-integrated spectral fitting technique
- **Gaia DR3 astrometry**: positions, parallaxes, proper motions, and their uncertainties for both components
- **Gaia FLAME masses** (`mass_flame_a`, `mass_flame_b`) with upper/lower confidence bounds
- **Instrument** used for each pair (Magellan/MIKE, VLT/UVES)
- **Gaia RVs** (where available) for consistency checks

Of the 85 pairs, 57 satisfy our bound selection criterion (scaled velocity $\tilde{v} < 2.5$) and are used in the MOND analysis.

## Analysis

`mond_analysis.py` is the main analysis script, containing three analyses in a single clean file:

1. **Bound system selection** — Computes scaled velocity $\tilde{v}$ and selects 57 bound systems ($\tilde{v} < 2.5$)
2. **MOND $a_0$ inference** — Hierarchical Bayesian model with EFE, run for both interpolating functions ($b=1,2$) and three prior ranges
3. **Supplementary $\gamma$ test** — Fits $G_{\rm eff} = \gamma G_{\rm N}$ (Appendix C of paper)

The `notebooks/` directory contains the full development notebook with exploratory code and plotting scripts.

### Dependencies

```
numpy
pandas
matplotlib
pymc >= 5.0
arviz
pytensor
corner
```

Install with:
```bash
pip install numpy pandas matplotlib pymc arviz pytensor corner
```

## Running the Analysis

1. Clone this repository:
   ```bash
   git clone https://github.com/seratsaad/wb3d-mond.git
   cd wb3d-mond
   ```

2. Install dependencies (see above)

3. Run the analysis:
   ```bash
   python mond_analysis.py
   ```

Note: The full HMC inference (4 chains × 5000 steps) takes several hours on a modern CPU. 

## Citation

If you use this data or code, please cite:

```bibtex
@article{Saad2026,
  author  = {Saad, Serat M. and Ting, Yuan-Sen},
  title   = {High-Precision Differential Radial Velocities of C3PO Wide Binaries: A Test of Modified Newtonian Dynamics (MOND)},
  journal = {arXiv e-prints},
  year    = {2026},
  eprint  = {XXXX.XXXXX},
}
```

## License

This project is released under the MIT License.
