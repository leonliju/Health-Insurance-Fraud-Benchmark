#!/usr/bin/env python3
"""
Generate a CLEAN (fraud-free) synthetic health-insurance claims dataset.

This is the fallback / contract generator for the project. It does not use
Synthea. It produces the shared claims table (17 columns, see README) plus
reference tables that the fraud-injection engine will use later.

All parameters live in config/params.json and are ILLUSTRATIVE placeholders
until recalibrated against published Indian statistics.

Usage:
    python src/generate_claims.py --config config/params.json --out data
    python src/generate_claims.py --n-patients 5000 --seed 7 --out data_small
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

CLAIM_COLUMNS = [
    "claim_id", "patient_id", "hospital_id", "doctor_id",
    "admission_date", "discharge_date",
    "patient_age", "patient_sex", "patient_district",
    "diagnosis_code", "procedure_package_code",
    "claimed_amount", "reference_package_rate", "length_of_stay",
    "is_fraud", "fraud_type", "severity_tier",
]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Reference entities
# ----------------------------------------------------------------------
def build_hospitals(cfg, rng, specialties):
    n = cfg["n_hospitals"]
    n_dist = cfg["n_districts"]
    district_weights = rng.lognormal(0, 0.5, n_dist)
    district_weights /= district_weights.sum()
    district = rng.choice(n_dist, size=n, p=district_weights)
    h_type = np.where(rng.random(n) < cfg["share_public_hospitals"], "public", "private")
    size = rng.lognormal(0, 0.8, n)
    size_norm = size / size.mean()

    offered = []
    for i in range(n):
        k = int(np.clip(1 + rng.poisson(1 + 1.5 * size_norm[i]), 1, len(specialties)))
        others = [s for s in specialties if s != "General Medicine"]
        chosen = list(rng.choice(others, size=min(k - 1, len(others)), replace=False)) if k > 1 else []
        offered.append(["General Medicine"] + chosen)

    # guarantee every specialty is offered somewhere
    biggest = int(np.argmax(size))
    for s in specialties:
        if not any(s in o for o in offered):
            offered[biggest].append(s)

    hospitals = pd.DataFrame({
        "hospital_id": [f"H{i + 1:03d}" for i in range(n)],
        "district_idx": district,
        "district": [f"District_{d + 1:03d}" for d in district],
        "hospital_type": h_type,
        "size_weight": np.round(size, 4),
        "specialties": ["|".join(o) for o in offered],
    })
    return hospitals, district_weights


def build_doctors(cfg, rng, hospitals):
    rows = []
    d = 0
    size = hospitals["size_weight"].to_numpy()
    size_norm = size / size.mean()
    for i, h in hospitals.iterrows():
        for spec in h["specialties"].split("|"):
            n_doc = 1 + rng.poisson(cfg["doctor_extra_lambda"] * size_norm[i])
            for _ in range(int(n_doc)):
                d += 1
                rows.append((f"D{d:04d}", h["hospital_id"], i, spec, rng.lognormal(0, 0.5)))
    return pd.DataFrame(rows, columns=["doctor_id", "hospital_id", "hospital_idx", "specialty", "workload_weight"])


def build_patients(cfg, rng, district_weights):
    n = cfg["n_patients"]
    buckets = cfg["age_buckets"]
    shares = np.array([b["share"] for b in buckets], dtype=float)
    shares /= shares.sum()
    bucket_idx = rng.choice(len(buckets), size=n, p=shares)
    lo = np.array([b["min"] for b in buckets])[bucket_idx]
    hi = np.array([b["max"] for b in buckets])[bucket_idx]
    age_at_start = rng.integers(lo, hi + 1)
    start = pd.Timestamp(cfg["window_start"])
    dob = start - pd.to_timedelta((age_at_start * 365.25 + rng.integers(0, 365, n)).astype(int), unit="D")
    sex = np.where(rng.random(n) < cfg["female_share"], "F", "M")
    district_idx = rng.choice(cfg["n_districts"], size=n, p=district_weights)
    return pd.DataFrame({
        "patient_id": [f"P{i + 1:05d}" for i in range(n)],
        "dob": dob,
        "patient_sex": sex,
        "district_idx": district_idx,
        "patient_district": [f"District_{d + 1:03d}" for d in district_idx],
    })


# ----------------------------------------------------------------------
# Claim generation
# ----------------------------------------------------------------------
def choose_packages(cfg, rng, patients, packages):
    """Pick a package for each admission using prevalence weight x age fit x sex rule."""
    start = pd.Timestamp(cfg["window_start"])
    end = pd.Timestamp(cfg["window_end"])
    mid = start + (end - start) / 2
    age_mid = ((mid - patients["dob"]).dt.days / 365.25).to_numpy()
    sex = patients["patient_sex"].to_numpy()

    probs = np.zeros((len(patients), len(packages)))
    for j, p in packages.iterrows():
        # normal density of age given package (divided by the spread so that
        # broad-age packages do not dominate narrow-age ones)
        fit = np.exp(-0.5 * ((age_mid - p["age_mean"]) / p["age_sd"]) ** 2) / p["age_sd"]
        mask = np.ones(len(patients)) if p["sex"] == "any" else (sex == p["sex"]).astype(float)
        probs[:, j] = p["weight"] * fit * mask
    probs += 1e-12
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def generate(cfg, seed=None):
    seed = cfg["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    packages = pd.DataFrame(cfg["packages"]).reset_index(drop=True)
    packages = packages.rename(columns={"reference_rate": "reference_package_rate"})
    specialties = list(dict.fromkeys(packages["specialty"].tolist()))
    start = pd.Timestamp(cfg["window_start"])
    end = pd.Timestamp(cfg["window_end"])
    window_days = (end - start).days

    hospitals, district_weights = build_hospitals(cfg, rng, specialties)
    doctors = build_doctors(cfg, rng, hospitals)
    patients = build_patients(cfg, rng, district_weights)

    # --- admissions per patient and package choice ---
    n_claims = 1 + rng.poisson(cfg["extra_claims_lambda"], len(patients))
    pat_idx = np.repeat(np.arange(len(patients)), n_claims)
    probs = choose_packages(cfg, rng, patients, packages)
    cum = probs[pat_idx].cumsum(axis=1)
    u = rng.random((len(pat_idx), 1))
    pkg_idx = np.minimum((cum < u).sum(axis=1), len(packages) - 1)

    n = len(pat_idx)
    claims = pd.DataFrame({
        "patient_idx": pat_idx,
        "pkg_idx": pkg_idx,
        "admission_date": start + pd.to_timedelta(rng.integers(0, window_days, n), unit="D"),
    })
    claims["specialty"] = packages["specialty"].to_numpy()[pkg_idx]
    claims["district_idx"] = patients["district_idx"].to_numpy()[pat_idx]

    # --- hospital choice: local first, otherwise any hospital offering the specialty ---
    hosp_offers = {s: hospitals.index[hospitals["specialties"].str.contains(s, regex=False)].to_numpy()
                   for s in specialties}
    hosp_size = hospitals["size_weight"].to_numpy()
    hosp_district = hospitals["district_idx"].to_numpy()
    claims["hospital_idx"] = -1
    for (dist, spec), grp in claims.groupby(["district_idx", "specialty"]):
        idx = grp.index.to_numpy()
        glob = hosp_offers[spec]
        loc = glob[hosp_district[glob] == dist]
        chosen = np.empty(len(idx), dtype=int)
        use_local = (rng.random(len(idx)) < cfg["local_hospital_share"]) & (len(loc) > 0)
        if use_local.any():
            w = hosp_size[loc] / hosp_size[loc].sum()
            chosen[use_local] = rng.choice(loc, size=int(use_local.sum()), p=w)
        if (~use_local).any():
            w = hosp_size[glob] / hosp_size[glob].sum()
            chosen[~use_local] = rng.choice(glob, size=int((~use_local).sum()), p=w)
        claims.loc[idx, "hospital_idx"] = chosen

    # --- doctor choice: within hospital and specialty ---
    claims["doctor_pos"] = -1
    doc_groups = doctors.groupby(["hospital_idx", "specialty"])
    doc_lookup = {k: (g.index.to_numpy(), g["workload_weight"].to_numpy()) for k, g in doc_groups}
    for (h, spec), grp in claims.groupby(["hospital_idx", "specialty"]):
        ids, w = doc_lookup[(h, spec)]
        claims.loc[grp.index, "doctor_pos"] = rng.choice(ids, size=len(grp), p=w / w.sum())

    # --- length of stay ---
    los_mean = packages["los_mean"].to_numpy()[claims["pkg_idx"]]
    los_sd = packages["los_sd"].to_numpy()[claims["pkg_idx"]]
    los = np.rint(rng.normal(los_mean, los_sd)).astype(int)
    claims["length_of_stay"] = np.clip(los, 1, np.ceil(los_mean + 3 * los_sd).astype(int))

    # --- repeat claims (chronic / cyclical treatment: dialysis, chemotherapy) ---
    base = claims.copy()
    extra_frames = []
    rep_mean = packages["repeat_mean"].to_numpy()[base["pkg_idx"]]
    rep_max = packages["repeat_max"].to_numpy()[base["pkg_idx"]]
    k = np.minimum(rng.poisson(rep_mean), rep_max)
    prev = base.loc[k > 0].copy()
    prev["_k"] = k[k > 0]
    j = 0
    while len(prev) > 0:
        j += 1
        prev = prev[prev["_k"] >= j].copy()
        if prev.empty:
            break
        gmin = packages["gap_min"].to_numpy()[prev["pkg_idx"]]
        gmax = packages["gap_max"].to_numpy()[prev["pkg_idx"]]
        gap = rng.integers(gmin, gmax + 1)
        nxt = prev.copy()
        nxt["admission_date"] = prev["admission_date"] + pd.to_timedelta(prev["length_of_stay"] + gap, unit="D")
        los_m = packages["los_mean"].to_numpy()[nxt["pkg_idx"]]
        los_s = packages["los_sd"].to_numpy()[nxt["pkg_idx"]]
        nxt["length_of_stay"] = np.clip(np.rint(rng.normal(los_m, los_s)).astype(int), 1, np.ceil(los_m + 3 * los_s).astype(int))
        nxt = nxt[nxt["admission_date"] <= end]
        extra_frames.append(nxt)
        prev = nxt
    if extra_frames:
        claims = pd.concat([claims] + [f.drop(columns="_k") for f in extra_frames], ignore_index=True)

    # --- discharge, amounts, ages ---
    claims["discharge_date"] = claims["admission_date"] + pd.to_timedelta(claims["length_of_stay"], unit="D")
    ref = packages["reference_package_rate"].to_numpy()[claims["pkg_idx"]]
    ratio = np.clip(rng.lognormal(0, cfg["amount_ratio_sigma"], len(claims)),
                    cfg["amount_ratio_clip"][0], cfg["amount_ratio_clip"][1])
    claims["reference_package_rate"] = ref
    claims["claimed_amount"] = (np.rint(ref * ratio / 10) * 10).astype(int)
    dob = patients["dob"].to_numpy()[claims["patient_idx"]]
    claims["patient_age"] = ((claims["admission_date"].to_numpy() - dob).astype("timedelta64[D]").astype(int) / 365.25).astype(int)

    # --- final ids, labels, column order ---
    claims = claims.sort_values(["admission_date", "patient_idx"]).reset_index(drop=True)
    claims["claim_id"] = [f"C{i + 1:06d}" for i in range(len(claims))]
    claims["patient_id"] = patients["patient_id"].to_numpy()[claims["patient_idx"]]
    claims["patient_sex"] = patients["patient_sex"].to_numpy()[claims["patient_idx"]]
    claims["patient_district"] = patients["patient_district"].to_numpy()[claims["patient_idx"]]
    claims["hospital_id"] = hospitals["hospital_id"].to_numpy()[claims["hospital_idx"]]
    claims["doctor_id"] = doctors["doctor_id"].to_numpy()[claims["doctor_pos"]]
    claims["diagnosis_code"] = packages["diagnosis_code"].to_numpy()[claims["pkg_idx"]]
    claims["procedure_package_code"] = packages["package_code"].to_numpy()[claims["pkg_idx"]]
    claims["is_fraud"] = 0
    claims["fraud_type"] = "none"
    claims["severity_tier"] = "none"
    claims["admission_date"] = claims["admission_date"].dt.strftime("%Y-%m-%d")
    claims["discharge_date"] = claims["discharge_date"].dt.strftime("%Y-%m-%d")
    claims = claims[CLAIM_COLUMNS]

    patients_out = patients[["patient_id", "patient_sex", "patient_district"]].copy()
    patients_out["date_of_birth"] = patients["dob"].dt.strftime("%Y-%m-%d")
    hospitals_out = hospitals[["hospital_id", "district", "hospital_type", "size_weight", "specialties"]]
    doctors_out = doctors[["doctor_id", "hospital_id", "specialty", "workload_weight"]]
    packages_out = packages[["package_code", "package_name", "diagnosis_code", "specialty",
                             "reference_package_rate", "sex", "los_mean", "repeat_mean"]]
    return claims, patients_out, hospitals_out, doctors_out, packages_out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/params.json")
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-patients", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.n_patients:
        cfg["n_patients"] = args.n_patients
    claims, patients, hospitals, doctors, packages = generate(cfg, args.seed)

    os.makedirs(args.out, exist_ok=True)
    claims.to_csv(os.path.join(args.out, "claims_clean.csv"), index=False)
    patients.to_csv(os.path.join(args.out, "patients.csv"), index=False)
    hospitals.to_csv(os.path.join(args.out, "hospitals.csv"), index=False)
    doctors.to_csv(os.path.join(args.out, "doctors.csv"), index=False)
    packages.to_csv(os.path.join(args.out, "packages.csv"), index=False)
    print(f"Wrote {len(claims):,} claims, {len(patients):,} patients, "
          f"{len(hospitals):,} hospitals, {len(doctors):,} doctors to {args.out}/")


if __name__ == "__main__":
    main()
