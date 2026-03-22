#!/usr/bin/env python3
"""
=============================================================================
MOND Wide Binary Analysis — Saad & Ting (2026)
=============================================================================

Hierarchical Bayesian inference for the MOND acceleration scale a_0
using high-precision differential radial velocities of C3PO wide binaries.

This script contains three analyses:
  1. Bound system selection (v_tilde < 2.5)
  2. MOND a_0 inference with EFE (b=1 and b=2, multiple priors)
  3. Supplementary gamma test (G_eff = gamma * G_N)

Usage:
    python mond_analysis.py

Requirements: numpy, pandas, pymc, pytensor, arviz, matplotlib, scipy
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from scipy import stats

# =============================================================================
# CONSTANTS
# =============================================================================
AU = 1.496e11              # meters
PC_TO_M = 3.085677581e16   # meters per parsec
G = 6.67430e-11            # m^3 kg^-1 s^-2
M_sun = 1.98847e30         # kg
K_PM = 4.74047             # km/s per mas/yr per kpc
A_EXT = 2.1e-10            # External field strength (m/s^2), solar neighborhood

# =============================================================================
# CONFIGURATION
# =============================================================================
CSV_PATH = "data/c3po_wide_binaries.csv"
OUTPUT_DIR = "results"

# MCMC settings
N_DRAWS = 2000
N_TUNE = 3000
N_CHAINS = 4
TARGET_ACCEPT = 0.95
MAX_TREEDEPTH = 10
RANDOM_SEED = 42

# Prior ranges to test: (lower, upper) for log10(a0)
PRIOR_RANGES = [(-13, -7), (-12, -8), (-11, -9)]

# Interpolating function choices
B_VALUES = [1, 2]

# Semi-major axis prior width
A_PRIOR_SIGMA = 0.6


# =============================================================================
# STEP 1: BOUND SYSTEM SELECTION (v_tilde < 2.5)
# =============================================================================
def compute_vtilde(df, cutoff=2.5):
    """
    Compute scaled velocity v_tilde = v_sky / sqrt(G M_tot / r_sky)
    and select bound systems with v_tilde < cutoff.

    Following the framework of Cookson et al. (2026).
    """
    # Mean parallax and distance
    parallax_mean = 0.5 * (df["parallax_a"] + df["parallax_b"])
    distance_pc = 1000.0 / parallax_mean
    distance_m = distance_pc * PC_TO_M

    # Proper motion difference (Gaia pmra already includes cos(dec))
    dpmra = df["pmra_a"] - df["pmra_b"]
    dpmdec = df["pmdec_a"] - df["pmdec_b"]
    mu_rel = np.sqrt(dpmra**2 + dpmdec**2)  # mas/yr

    # Sky-projected velocity: v(km/s) = 4.74047 * mu("/yr) * d(pc)
    v_sky_ms = K_PM * (mu_rel / 1000.0) * distance_pc * 1000.0  # m/s

    # Angular separation -> projected separation
    ra_a, dec_a = np.deg2rad(df["ra_a"]), np.deg2rad(df["dec_a"])
    ra_b, dec_b = np.deg2rad(df["ra_b"]), np.deg2rad(df["dec_b"])
    cos_theta = (np.sin(dec_a) * np.sin(dec_b) +
                 np.cos(dec_a) * np.cos(dec_b) * np.cos(ra_a - ra_b))
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    r_sky_m = theta_rad * distance_m

    # Total mass
    M_tot_kg = df["mass_total_flame"].values * M_sun

    # Scaled velocity
    v_newton = np.sqrt(G * M_tot_kg / r_sky_m)
    v_tilde = v_sky_ms.values / v_newton

    # Selection
    is_bound = (v_tilde < cutoff) & np.isfinite(v_tilde)

    print(f"  Total systems: {len(df)}")
    print(f"  Bound systems (v_tilde < {cutoff}): {is_bound.sum()}")

    return df[is_bound].reset_index(drop=True)


# =============================================================================
# DATA PREPARATION
# =============================================================================
def prepare_data(csv_path, vtilde_cut=2.5, frac_floor=0.05):
    """
    Load CSV, require FLAME masses, apply v_tilde bound selection,
    and compute all quantities needed for the orbital model.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} systems")

    # Require FLAME masses
    df = df.dropna(subset=["mass_flame_a", "mass_flame_b"]).copy()
    print(f"After FLAME mass cut: {len(df)} systems")

    # Apply v_tilde bound selection
    df = compute_vtilde(df, cutoff=vtilde_cut)
    N = len(df)
    print(f"Final sample: {N} systems")

    # Distances
    d_a_pc = 1000.0 / df["parallax_a"].values
    d_b_pc = 1000.0 / df["parallax_b"].values
    d_mean_pc = 0.5 * (d_a_pc + d_b_pc)

    # Coordinates in radians
    ra_a = np.deg2rad(df["ra_a"].values)
    dec_a = np.deg2rad(df["dec_a"].values)
    ra_b = np.deg2rad(df["ra_b"].values)
    dec_b = np.deg2rad(df["dec_b"].values)

    # Projected separation
    delta_ra = (ra_b - ra_a) * np.cos(0.5 * (dec_a + dec_b))
    delta_dec = dec_b - dec_a
    sep_rad = np.sqrt(delta_ra**2 + delta_dec**2)
    r_obs = d_mean_pc * sep_rad * PC_TO_M
    r_proj_au = r_obs / AU

    # Separation uncertainty (5% floor)
    par_err_a = df["parallax_error_a"].values
    par_err_b = df["parallax_error_b"].values
    d_a_err = 1000.0 * par_err_a / (df["parallax_a"].values**2) * PC_TO_M
    d_b_err = 1000.0 * par_err_b / (df["parallax_b"].values**2) * PC_TO_M
    d_mean_err = 0.5 * np.sqrt(d_a_err**2 + d_b_err**2)
    r_err = np.maximum(frac_floor * r_obs, d_mean_err)

    # Differential proper motions
    dpmra = df["pmra_b"].values - df["pmra_a"].values
    dpmdec = df["pmdec_b"].values - df["pmdec_a"].values
    pm_err_a = np.sqrt(df["pmra_error_a"].values**2 + df["pmdec_error_a"].values**2)
    pm_err_b = np.sqrt(df["pmra_error_b"].values**2 + df["pmdec_error_b"].values**2)
    pm_err = 0.5 * np.sqrt(pm_err_a**2 + pm_err_b**2)

    # Differential RV (CSV is in km/s, convert to m/s)
    rv_diff_ms = df["rv_diff"].values * 1000.0
    rv_sigma_ms = df["rv_sigma"].values * 1000.0

    # FLAME masses and uncertainties
    M1_flame = df["mass_flame_a"].values
    M2_flame = df["mass_flame_b"].values
    M1_err = df["mass_flame_a_err"].values
    M2_err = df["mass_flame_b_err"].values

    return pd.DataFrame({
        "r_obs": r_obs, "r_err": r_err,
        "rv_diff": rv_diff_ms, "rv_sigma": rv_sigma_ms,
        "distance_pc": d_mean_pc,
        "distance_a_pc": d_a_pc, "distance_b_pc": d_b_pc,
        "ra_a": ra_a, "dec_a": dec_a, "ra_b": ra_b, "dec_b": dec_b,
        "pmra_diff": dpmra, "pmdec_diff": dpmdec, "pm_err": pm_err,
        "M1_flame": M1_flame, "M2_flame": M2_flame,
        "M1_err": M1_err, "M2_err": M2_err,
        "r_proj_au": r_proj_au,
    })


# =============================================================================
# STEP 2: MOND MODEL (a_0 inference with EFE)
# =============================================================================
def build_mond_model(data, a_ext_val, prior_range, b_value):
    """
    Hierarchical Bayesian model for MOND a_0 inference.

    Parameters
    ----------
    data : DataFrame from prepare_data()
    a_ext_val : float, external field strength (m/s^2)
    prior_range : tuple (lower, upper) for log10(a0) uniform prior
    b_value : int, interpolating function parameter (1=simple, 2=standard)
    """
    N = len(data)
    prior_lo, prior_hi = prior_range

    # Galactic center direction
    ra_gc = np.deg2rad(266.4051)
    dec_gc = np.deg2rad(-28.936175)
    gc_hat_np = np.array([
        np.cos(dec_gc) * np.cos(ra_gc),
        np.cos(dec_gc) * np.sin(ra_gc),
        np.sin(dec_gc)
    ], dtype="float64")

    with pm.Model() as model:
        # --- Data containers ---
        r_obs = pm.Data("r_obs", data["r_obs"].values)
        r_err = pm.Data("r_err", data["r_err"].values)
        RV_diff = pm.Data("RV_diff", data["rv_diff"].values)
        RV_sigma = pm.Data("RV_sigma", data["rv_sigma"].values)
        ra_a = pm.Data("ra_a_rad", data["ra_a"].values)
        dec_a = pm.Data("dec_a_rad", data["dec_a"].values)
        ra_b = pm.Data("ra_b_rad", data["ra_b"].values)
        dec_b = pm.Data("dec_b_rad", data["dec_b"].values)
        distance_a_pc = pm.Data("distance_a_pc", data["distance_a_pc"].values)
        distance_b_pc = pm.Data("distance_b_pc", data["distance_b_pc"].values)
        distance_pc = pm.Data("distance_pc", data["distance_pc"].values)

        # --- Stellar masses (FLAME priors) ---
        M1_sol = pm.Normal("M1_sol", mu=data["M1_flame"].values,
                           sigma=data["M1_err"].values, shape=N)
        M2_sol = pm.Normal("M2_sol", mu=data["M2_flame"].values,
                           sigma=data["M2_err"].values, shape=N)
        M1 = pm.Deterministic("M1", M1_sol * M_sun)
        M2 = pm.Deterministic("M2", M2_sol * M_sun)

        # --- Semi-major axis (eccentricity-dependent prior) ---
        # Eccentricity first (needed for SMA prior center)
        sep_au = pm.Data("sep_au", data["r_proj_au"].values)
        bin_edges = np.array([0, 100, 300, 1000, 3000, 1e6], dtype=float)
        bin_idx = np.clip(
            np.digitize(data["r_proj_au"].values, bin_edges) - 1,
            0, len(bin_edges) - 2
        )
        bin_idx_data = pm.Data("bin_idx", bin_idx)

        mu_alpha = np.array([-0.2, 0.4, 0.8, 1.0, 1.2])
        sigma_alpha = np.array([0.3, 0.2, 0.2, 0.2, 0.2])
        alpha_bins = pm.Normal("alpha_bins", mu=mu_alpha,
                               sigma=sigma_alpha, shape=len(mu_alpha))
        alpha = pm.Deterministic("alpha", alpha_bins[bin_idx_data])

        e_raw = pm.Beta("e_raw", alpha=pt.maximum(alpha + 1.0, 0.1),
                        beta=1.0, shape=N)
        e = pm.Deterministic("e", pt.minimum(e_raw, 0.98))

        # SMA prior: eta ~ N(mu_e, sigma_a), where
        # mu_e = sqrt(1-e^2) - ln(1 + sqrt(1-e^2))  [van Albada 1968]
        s_e = pt.sqrt(pt.clip(1.0 - e**2, 1e-12, np.inf))
        mu_eta = pm.Deterministic("mu_eta", s_e - pt.log(1.0 + s_e))
        eta = pm.Normal("eta", mu=mu_eta, sigma=A_PRIOR_SIGMA, shape=N)
        a = pm.Deterministic("a", r_obs * pt.exp(eta))

        # --- MOND acceleration scale ---
        log10a0 = pm.Uniform("log10a0", lower=prior_lo, upper=prior_hi)
        a0 = pm.Deterministic("a0", 10**log10a0)

        # --- Orbital angles ---
        M_anom = pm.Uniform("M_anom", 0.0, 2 * np.pi, shape=N)
        cos_i = pm.Uniform("cos_i", lower=-1.0, upper=1.0, shape=N)
        i = pm.Deterministic("i", pt.arccos(cos_i))
        Omega = pm.Uniform("Omega", lower=0.0, upper=2 * np.pi, shape=N)
        omega = pm.Uniform("omega", lower=0.0, upper=2 * np.pi, shape=N)

        # --- Kepler solver (Newton-Raphson, 7 iterations) ---
        def kepler_E(M, e, n_iter=7):
            E = M + e * pt.sin(M) + 0.5 * e**2 * pt.sin(2 * M)
            for _ in range(n_iter):
                f = E - e * pt.sin(E) - M
                fp = 1.0 - e * pt.cos(E)
                E = E - f / fp
            return E

        def nu_from_E(E, e):
            s = pt.sqrt((1.0 + e) / (1.0 - e))
            return 2.0 * pt.arctan2(s * pt.sin(E / 2.0), pt.cos(E / 2.0))

        E_val = kepler_E(M_anom, e)
        nu = pm.Deterministic("nu", nu_from_E(E_val, e))

        denom = pt.clip(1.0 + e * pt.cos(nu), 1e-9, np.inf)
        r_true = pm.Deterministic("r_true", a * (1.0 - e**2) / denom)

        # --- Interpolating function (Eq. 5 in paper) ---
        def mu_interp(x, b):
            if b == 1:
                return x / (1.0 + x)
            else:
                return x / pt.sqrt(1.0 + x**2)

        # --- MOND effective acceleration with EFE (Eq. 7 in paper) ---
        def mond_a_eff(aN, theta, a0_val, a_ext, b):
            a_tot = pt.sqrt(aN**2 + a_ext**2 + 2 * aN * a_ext * pt.cos(theta))
            mu_t = mu_interp(a_tot / a0_val, b)
            mu_e = mu_interp(a_ext / a0_val, b)
            inner = (mu_t**2 * aN**2
                     + (mu_t - mu_e)**2 * a_ext**2
                     + 2 * mu_t * (mu_t - mu_e) * aN * a_ext * pt.cos(theta))
            return pt.sqrt(inner)

        # --- Rotation matrices ---
        def rot_z(angle):
            c, s = pt.cos(angle), pt.sin(angle)
            z, o = pt.zeros_like(c), pt.ones_like(c)
            return pt.stack([
                pt.stack([c, -s, z], axis=-1),
                pt.stack([s, c, z], axis=-1),
                pt.stack([z, z, o], axis=-1),
            ], axis=-2)

        def rot_x(angle):
            c, s = pt.cos(angle), pt.sin(angle)
            z, o = pt.zeros_like(c), pt.ones_like(c)
            return pt.stack([
                pt.stack([o, z, z], axis=-1),
                pt.stack([z, c, -s], axis=-1),
                pt.stack([z, s, c], axis=-1),
            ], axis=-2)

        R = rot_z(Omega) @ rot_x(i) @ rot_z(omega)

        # --- Velocities ---
        F = (1.0 + e**2 + 2 * e * pt.cos(nu)) / (1.0 + e * pt.cos(nu))
        F = pt.clip(F, 1e-9, np.inf)

        aN_rel = G * (M1 + M2) / r_true**2

        # Position in ICRS (for EFE angle)
        r_orb = pt.stack([r_true * pt.cos(nu), r_true * pt.sin(nu),
                          pt.zeros_like(r_true)], axis=1)
        r_icrs = (R @ r_orb[..., None]).squeeze(-1)
        r_hat = r_icrs / pt.sqrt(pt.sum(r_icrs**2, axis=1, keepdims=True))

        gc_hat = pt.as_tensor_variable(gc_hat_np)[None, :]
        cos_theta_ext = pt.clip(pt.sum(r_hat * gc_hat, axis=1), -1.0, 1.0)
        theta_ext = pm.Deterministic("theta_ext", pt.arccos(cos_theta_ext))

        # MOND effective acceleration
        a_eff = mond_a_eff(aN_rel, theta_ext, a0, a_ext_val, b_value)

        v_rel = pt.sqrt(pt.maximum(a_eff * r_true * F, 1e-30))

        vr_fac = -e * pt.sin(nu) / pt.sqrt(1.0 + e**2 + 2 * e * pt.cos(nu))
        vt_fac = (1.0 + e * pt.cos(nu)) / pt.sqrt(1.0 + e**2 + 2 * e * pt.cos(nu))

        vx = v_rel * vr_fac * pt.cos(nu) - v_rel * vt_fac * pt.sin(nu)
        vy = v_rel * vr_fac * pt.sin(nu) + v_rel * vt_fac * pt.cos(nu)

        coef1 = M2 / (M1 + M2)
        coef2 = -M1 / (M1 + M2)
        v1_orb = pt.stack([coef1 * vx, coef1 * vy, pt.zeros_like(vx)], axis=1)
        v2_orb = pt.stack([coef2 * vx, coef2 * vy, pt.zeros_like(vx)], axis=1)

        v1_icrs = (R @ v1_orb[..., None]).squeeze(-1)
        v2_icrs = (R @ v2_orb[..., None]).squeeze(-1)

        # --- Coordinate triads ---
        def triad(ra, dec):
            cdec, sdec = pt.cos(dec), pt.sin(dec)
            cra, sra = pt.cos(ra), pt.sin(ra)
            r_h = pt.stack([cdec * cra, cdec * sra, sdec], axis=1)
            e_h = pt.stack([-sra, cra, pt.zeros_like(ra)], axis=1)
            n_h = pt.stack([-sdec * cra, -sdec * sra, cdec], axis=1)
            return r_h, e_h, n_h

        rhat_a, ehat_a, nhat_a = triad(ra_a, dec_a)
        rhat_b, ehat_b, nhat_b = triad(ra_b, dec_b)

        # --- Systemic velocity ---
        v_sys_x = pm.Normal("v_sys_x", mu=0.0, sigma=30e3, shape=N)
        v_sys_y = pm.Normal("v_sys_y", mu=0.0, sigma=30e3, shape=N)
        v_sys_z = pm.Normal("v_sys_z", mu=0.0, sigma=30e3, shape=N)
        v_sys = pt.stack([v_sys_x, v_sys_y, v_sys_z], axis=-1)

        # --- Model observables ---
        # Differential RV
        rv1 = pt.sum(v1_icrs * rhat_a, axis=1) + pt.sum(v_sys * rhat_a, axis=1)
        rv2 = pt.sum(v2_icrs * rhat_b, axis=1) + pt.sum(v_sys * rhat_b, axis=1)
        RV_diff_model = pm.Deterministic("RV_diff_model", rv2 - rv1)

        # Differential proper motions
        v1_t_a = v1_icrs - (pt.sum(v1_icrs * rhat_a, axis=1)[:, None]) * rhat_a
        v2_t_b = v2_icrs - (pt.sum(v2_icrs * rhat_b, axis=1)[:, None]) * rhat_b
        v_sys_t_a = v_sys - (pt.sum(v_sys * rhat_a, axis=1)[:, None]) * rhat_a
        v_sys_t_b = v_sys - (pt.sum(v_sys * rhat_b, axis=1)[:, None]) * rhat_b

        pmra_model = pm.Deterministic("pmra_diff_model",
            -(pt.sum((v2_t_b + v_sys_t_b) * ehat_b, axis=1) / (K_PM * distance_b_pc) -
              pt.sum((v1_t_a + v_sys_t_a) * ehat_a, axis=1) / (K_PM * distance_a_pc)))
        pmdec_model = pm.Deterministic("pmdec_diff_model",
            (pt.sum((v2_t_b + v_sys_t_b) * nhat_b, axis=1) / (K_PM * distance_b_pc) -
             pt.sum((v1_t_a + v_sys_t_a) * nhat_a, axis=1) / (K_PM * distance_a_pc)))

        # Projected separation
        r_E = pt.sum(r_icrs * ehat_a, axis=1)
        r_N = pt.sum(r_icrs * nhat_a, axis=1)
        r_proj_model = pm.Deterministic("r_proj_model", pt.sqrt(r_E**2 + r_N**2))

        # --- Jitter terms ---
        rv_jitter = pm.HalfNormal("rv_jitter", sigma=10.0)
        pm_jitter = pm.HalfNormal("pm_jitter", sigma=0.05)

        # --- Likelihoods ---
        pm.Normal("lik_rproj", mu=r_proj_model, sigma=r_err, observed=r_obs)
        pm.StudentT("lik_RVdiff", nu=5, mu=RV_diff_model,
                    sigma=pt.sqrt(RV_sigma**2 + rv_jitter**2),
                    observed=RV_diff)

        dpmra_obs = pm.Data("dpmra_obs", data["pmra_diff"].values)
        dpmdec_obs = pm.Data("dpmdec_obs", data["pmdec_diff"].values)
        pm_err_dat = pm.Data("pm_err", data["pm_err"].values)
        pm.Normal("lik_pmra", mu=pmra_model,
                  sigma=pt.sqrt(pm_err_dat**2 + pm_jitter**2),
                  observed=dpmra_obs)
        pm.Normal("lik_pmdec", mu=pmdec_model,
                  sigma=pt.sqrt(pm_err_dat**2 + pm_jitter**2),
                  observed=dpmdec_obs)

    return model


# =============================================================================
# STEP 3: GAMMA MODEL (supplementary test)
# =============================================================================
def build_gamma_model(data):
    """
    Hierarchical model for gravity boost factor gamma = G_eff / G_N.
    Identical to MOND model but replaces MOND acceleration with gamma * G * M / r^2.
    See Appendix C of the paper.
    """
    N = len(data)

    with pm.Model() as model:
        # Data containers (same as MOND model)
        r_obs = pm.Data("r_obs", data["r_obs"].values)
        r_err = pm.Data("r_err", data["r_err"].values)
        RV_diff = pm.Data("RV_diff", data["rv_diff"].values)
        RV_sigma = pm.Data("RV_sigma", data["rv_sigma"].values)
        ra_a = pm.Data("ra_a_rad", data["ra_a"].values)
        dec_a = pm.Data("dec_a_rad", data["dec_a"].values)
        ra_b = pm.Data("ra_b_rad", data["ra_b"].values)
        dec_b = pm.Data("dec_b_rad", data["dec_b"].values)
        distance_a_pc = pm.Data("distance_a_pc", data["distance_a_pc"].values)
        distance_b_pc = pm.Data("distance_b_pc", data["distance_b_pc"].values)
        distance_pc = pm.Data("distance_pc", data["distance_pc"].values)

        # Global parameter: Gamma = log10(sqrt(gamma))
        Gam = pm.Uniform("Gam", lower=-1.0, upper=1.0)
        gamma = pm.Deterministic("gamma", 10.0 ** (2.0 * Gam))

        # Masses (log-normal around FLAME)
        M1_sol = pm.LogNormal("M1_sol", mu=np.log(data["M1_flame"].values),
                              sigma=0.05, shape=N)
        M2_sol = pm.LogNormal("M2_sol", mu=np.log(data["M2_flame"].values),
                              sigma=0.05, shape=N)
        M1 = M1_sol * M_sun
        M2 = M2_sol * M_sun

        # Eccentricity
        sep_au = pm.Data("sep_au", data["r_obs"].values / AU)
        bin_edges = np.array([0, 100, 300, 1000, 3000, 1e6], dtype=float)
        bin_idx = np.clip(
            np.digitize(np.asarray(sep_au.get_value()), bin_edges) - 1,
            0, len(bin_edges) - 2
        )
        bin_idx_data = pm.Data("bin_idx", bin_idx)

        mu_alpha = np.array([-0.2, 0.4, 0.8, 1.0, 1.2])
        sigma_alpha = np.array([0.3, 0.2, 0.2, 0.2, 0.2])
        alpha_bins = pm.Normal("alpha_bins", mu=mu_alpha,
                               sigma=sigma_alpha, shape=len(mu_alpha))
        alpha = pm.Deterministic("alpha", alpha_bins[bin_idx_data])

        e_raw = pm.Beta("e_raw", alpha=pt.maximum(alpha + 1.0, 0.1),
                        beta=1.0, shape=N)
        e = pm.Deterministic("e", pt.minimum(e_raw, 0.98))

        # Semi-major axis (eccentricity-dependent prior)
        s_e = pt.sqrt(pt.clip(1.0 - e**2, 1e-12, np.inf))
        mu_eta = s_e - pt.log(1.0 + s_e)
        eta = pm.Normal("eta", mu=mu_eta, sigma=A_PRIOR_SIGMA, shape=N)
        a = pm.Deterministic("a", r_obs * pt.exp(eta))

        # Orbital angles
        M_anom = pm.Uniform("M_anom", 0.0, 2 * np.pi, shape=N)
        cos_i = pm.Uniform("cos_i", lower=-1.0, upper=1.0, shape=N)
        i = pm.Deterministic("i", pt.arccos(cos_i))
        Omega = pm.Uniform("Omega", lower=0.0, upper=2 * np.pi, shape=N)
        omega = pm.Uniform("omega", lower=0.0, upper=2 * np.pi, shape=N)

        # Kepler solver
        def kepler_E(M, e, n_iter=7):
            E = M + e * pt.sin(M) + 0.5 * e**2 * pt.sin(2 * M)
            for _ in range(n_iter):
                E = E - (E - e * pt.sin(E) - M) / (1.0 - e * pt.cos(E))
            return E

        def nu_from_E(E, e):
            s = pt.sqrt((1.0 + e) / (1.0 - e))
            return 2.0 * pt.arctan2(s * pt.sin(E / 2.0), pt.cos(E / 2.0))

        E_val = kepler_E(M_anom, e)
        nu = pm.Deterministic("nu", nu_from_E(E_val, e))

        denom = pt.clip(1.0 + e * pt.cos(nu), 1e-9, np.inf)
        r_true = pm.Deterministic("r_true", a * (1.0 - e**2) / denom)

        # Rotation matrices
        def rot_z(angle):
            c, s = pt.cos(angle), pt.sin(angle)
            z, o = pt.zeros_like(c), pt.ones_like(c)
            return pt.stack([
                pt.stack([c, -s, z], axis=-1),
                pt.stack([s, c, z], axis=-1),
                pt.stack([z, z, o], axis=-1),
            ], axis=-2)

        def rot_x(angle):
            c, s = pt.cos(angle), pt.sin(angle)
            z, o = pt.zeros_like(c), pt.ones_like(c)
            return pt.stack([
                pt.stack([o, z, z], axis=-1),
                pt.stack([z, c, -s], axis=-1),
                pt.stack([z, s, c], axis=-1),
            ], axis=-2)

        R = rot_z(Omega) @ rot_x(i) @ rot_z(omega)

        # Velocity: v = sqrt(gamma * G * M_tot / r * F)
        F = pt.clip(
            (1.0 + e**2 + 2 * e * pt.cos(nu)) / (1.0 + e * pt.cos(nu)),
            1e-9, np.inf
        )
        v_rel = pt.sqrt(pt.maximum(gamma * G * (M1 + M2) / r_true * F, 1e-30))

        vr_fac = -e * pt.sin(nu) / pt.sqrt(1.0 + e**2 + 2 * e * pt.cos(nu))
        vt_fac = (1.0 + e * pt.cos(nu)) / pt.sqrt(1.0 + e**2 + 2 * e * pt.cos(nu))

        vx = v_rel * vr_fac * pt.cos(nu) - v_rel * vt_fac * pt.sin(nu)
        vy = v_rel * vr_fac * pt.sin(nu) + v_rel * vt_fac * pt.cos(nu)

        coef1 = M2 / (M1 + M2)
        coef2 = -M1 / (M1 + M2)
        v1_icrs = (R @ pt.stack([coef1*vx, coef1*vy, pt.zeros_like(vx)], axis=1)[..., None]).squeeze(-1)
        v2_icrs = (R @ pt.stack([coef2*vx, coef2*vy, pt.zeros_like(vx)], axis=1)[..., None]).squeeze(-1)

        # Triads and observables (same structure as MOND model)
        def triad(ra, dec):
            cdec, sdec = pt.cos(dec), pt.sin(dec)
            cra, sra = pt.cos(ra), pt.sin(ra)
            return (pt.stack([cdec*cra, cdec*sra, sdec], axis=1),
                    pt.stack([-sra, cra, pt.zeros_like(ra)], axis=1),
                    pt.stack([-sdec*cra, -sdec*sra, cdec], axis=1))

        rhat_a, ehat_a, nhat_a = triad(ra_a, dec_a)
        rhat_b, ehat_b, nhat_b = triad(ra_b, dec_b)

        v_sys_x = pm.Normal("v_sys_x", mu=0.0, sigma=30e3, shape=N)
        v_sys_y = pm.Normal("v_sys_y", mu=0.0, sigma=30e3, shape=N)
        v_sys_z = pm.Normal("v_sys_z", mu=0.0, sigma=30e3, shape=N)
        v_sys = pt.stack([v_sys_x, v_sys_y, v_sys_z], axis=-1)

        rv1 = pt.sum(v1_icrs * rhat_a, axis=1) + pt.sum(v_sys * rhat_a, axis=1)
        rv2 = pt.sum(v2_icrs * rhat_b, axis=1) + pt.sum(v_sys * rhat_b, axis=1)
        RV_model = pm.Deterministic("RV_diff_model", rv2 - rv1)

        r_orb = pt.stack([r_true*pt.cos(nu), r_true*pt.sin(nu), pt.zeros_like(r_true)], axis=1)
        r_icrs = (R @ r_orb[..., None]).squeeze(-1)
        r_proj_model = pm.Deterministic("r_proj_model",
            pt.sqrt(pt.sum(r_icrs * ehat_a, axis=1)**2 + pt.sum(r_icrs * nhat_a, axis=1)**2))

        # PM model
        v1_t = v1_icrs - (pt.sum(v1_icrs*rhat_a, axis=1)[:, None])*rhat_a + v_sys - (pt.sum(v_sys*rhat_a, axis=1)[:, None])*rhat_a
        v2_t = v2_icrs - (pt.sum(v2_icrs*rhat_b, axis=1)[:, None])*rhat_b + v_sys - (pt.sum(v_sys*rhat_b, axis=1)[:, None])*rhat_b

        pmra_model = pm.Deterministic("pmra_diff_model",
            -(pt.sum(v2_t*ehat_b, axis=1)/(K_PM*distance_b_pc) - pt.sum(v1_t*ehat_a, axis=1)/(K_PM*distance_a_pc)))
        pmdec_model = pm.Deterministic("pmdec_diff_model",
            (pt.sum(v2_t*nhat_b, axis=1)/(K_PM*distance_b_pc) - pt.sum(v1_t*nhat_a, axis=1)/(K_PM*distance_a_pc)))

        # Jitter and likelihoods
        rv_jitter = pm.HalfNormal("rv_jitter", sigma=10.0)
        pm_jitter = pm.HalfNormal("pm_jitter", sigma=0.05)

        pm.Normal("lik_rproj", mu=r_proj_model, sigma=r_err, observed=r_obs)
        pm.StudentT("lik_RVdiff", nu=5, mu=RV_model,
                    sigma=pt.sqrt(RV_sigma**2 + rv_jitter**2), observed=RV_diff)

        dpmra_obs = pm.Data("dpmra_obs", data["pmra_diff"].values)
        dpmdec_obs = pm.Data("dpmdec_obs", data["pmdec_diff"].values)
        pm_err_dat = pm.Data("pm_err", data["pm_err"].values)
        pm.Normal("lik_pmra", mu=pmra_model,
                  sigma=pt.sqrt(pm_err_dat**2 + pm_jitter**2), observed=dpmra_obs)
        pm.Normal("lik_pmdec", mu=pmdec_model,
                  sigma=pt.sqrt(pm_err_dat**2 + pm_jitter**2), observed=dpmdec_obs)

    return model


# =============================================================================
# RESULTS SUMMARY
# =============================================================================
def summarize_a0(trace, prior_range, b_value):
    """Compute posterior statistics for log10(a0)."""
    samples = trace.posterior["log10a0"].values.flatten()
    canonical = -9.92
    median = np.median(samples)
    ci_68 = np.percentile(samples, [16, 84])
    cdf = np.mean(samples < canonical)
    sigma = stats.norm.ppf(cdf) if cdf > 0 else 0

    print(f"  b={b_value}, prior=U{prior_range}:")
    print(f"    Median log10(a0) = {median:.2f}")
    print(f"    68% CI = [{ci_68[0]:.2f}, {ci_68[1]:.2f}]")
    print(f"    CDF at canonical = {cdf*100:.1f}%  ({sigma:.1f}sigma)")
    return {"b": b_value, "prior": prior_range, "median": median,
            "ci68": ci_68, "cdf": cdf, "sigma": sigma}


def summarize_gamma(trace):
    """Compute posterior statistics for gamma."""
    g = trace.posterior["gamma"].values.flatten()
    med = np.median(g)
    lo, hi = np.percentile(g, [16, 84])
    print(f"  gamma = {med:.2f} [{lo:.2f}, {hi:.2f}] (68% CI)")
    print(f"  Consistent with Newtonian (gamma=1): "
          f"{abs(med-1)/(hi-med):.1f}sigma")


# =============================================================================
# MAIN
# =============================================================================
def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Prepare data
    print("=" * 60)
    print("PREPARING DATA")
    print("=" * 60)
    data = prepare_data(CSV_PATH)

    # Run MOND analysis for all configurations
    print("\n" + "=" * 60)
    print("MOND a_0 INFERENCE")
    print("=" * 60)
    results = []
    for prior_range, b_value in product(PRIOR_RANGES, B_VALUES):
        prior_str = f"{prior_range[0]}to{prior_range[1]}"
        prefix = f"{OUTPUT_DIR}/mond_b{b_value}_prior{prior_str}"

        model = build_mond_model(data, A_EXT, prior_range, b_value)
        with model:
            trace = pm.sample(N_DRAWS, tune=N_TUNE, chains=N_CHAINS,
                              target_accept=TARGET_ACCEPT,
                              max_treedepth=MAX_TREEDEPTH,
                              return_inferencedata=True,
                              random_seed=RANDOM_SEED)
        trace.to_netcdf(f"{prefix}_trace.nc")
        results.append(summarize_a0(trace, prior_range, b_value))

    # Run gamma test
    print("\n" + "=" * 60)
    print("GAMMA TEST (Appendix C)")
    print("=" * 60)
    gamma_model = build_gamma_model(data)
    with gamma_model:
        gamma_trace = pm.sample(N_DRAWS, tune=N_TUNE, chains=N_CHAINS,
                                target_accept=0.95,
                                return_inferencedata=True,
                                random_seed=RANDOM_SEED)
    gamma_trace.to_netcdf(f"{OUTPUT_DIR}/gamma_trace.nc")
    summarize_gamma(gamma_trace)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"{'b':>3} | {'Prior':>12} | {'Median':>8} | {'CDF%':>6} | {'sigma':>6}")
    print("-" * 50)
    for r in results:
        print(f"{r['b']:>3} | U{r['prior']} | {r['median']:>8.2f} | "
              f"{r['cdf']*100:>5.1f}% | {r['sigma']:>5.1f}")


if __name__ == "__main__":
    main()
