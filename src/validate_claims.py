#!/usr/bin/env python3
"""
Sanity checks for the clean synthetic claims dataset.

Usage:
    python src/validate_claims.py --data data
Exits with status 1 if any hard check fails.
"""
import argparse
import os
import sys

import pandas as pd

EXPECTED_COLUMNS = [
    "claim_id", "patient_id", "hospital_id", "doctor_id",
    "admission_date", "discharge_date",
    "patient_age", "patient_sex", "patient_district",
    "diagnosis_code", "procedure_package_code",
    "claimed_amount", "reference_package_rate", "length_of_stay",
    "is_fraud", "fraud_type", "severity_tier",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--window-start", default="2023-01-01")
    ap.add_argument("--window-end", default="2025-12-31")
    args = ap.parse_args()

    claims = pd.read_csv(os.path.join(args.data, "claims_clean.csv"))
    hospitals = pd.read_csv(os.path.join(args.data, "hospitals.csv"))
    doctors = pd.read_csv(os.path.join(args.data, "doctors.csv"))
    packages = pd.read_csv(os.path.join(args.data, "packages.csv"))

    failures = []

    def check(name, ok):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failures.append(name)

    adm = pd.to_datetime(claims["admission_date"])
    dis = pd.to_datetime(claims["discharge_date"])

    check("columns match the shared schema (17, in order)", list(claims.columns) == EXPECTED_COLUMNS)
    check("no missing values", int(claims.isna().sum().sum()) == 0)
    check("claim_id is unique", claims["claim_id"].is_unique)
    check("admission dates inside the window",
          adm.min() >= pd.Timestamp(args.window_start) and adm.max() <= pd.Timestamp(args.window_end))
    check("discharge is after admission", bool((dis > adm).all()))
    check("length_of_stay equals discharge minus admission", bool(((dis - adm).dt.days == claims["length_of_stay"]).all()))
    check("length_of_stay >= 1", bool((claims["length_of_stay"] >= 1).all()))
    check("ages between 0 and 95", bool(claims["patient_age"].between(0, 95).all()))
    check("claimed_amount > 0", bool((claims["claimed_amount"] > 0).all()))
    ratio = claims["claimed_amount"] / claims["reference_package_rate"]
    check("genuine claim ratio within 0.85 to 1.15 (allowing rounding)", bool(ratio.between(0.849, 1.151).all()))
    check("clean file: is_fraud is all 0", bool((claims["is_fraud"] == 0).all()))
    check("clean file: fraud_type and severity_tier are 'none'",
          bool((claims["fraud_type"] == "none").all() and (claims["severity_tier"] == "none").all()))

    # sex rules and diagnosis-package pairing
    m = claims.merge(packages, left_on="procedure_package_code", right_on="package_code", how="left", suffixes=("", "_pkg"))
    check("every package code exists in packages.csv", bool(m["package_name"].notna().all()))
    check("diagnosis code matches the package", bool((m["diagnosis_code"] == m["diagnosis_code_pkg"]).all()))
    ok_sex = ((m["sex"] == "any") | (m["sex"] == m["patient_sex"])).all()
    check("sex restrictions respected (e.g. obstetrics = female)", bool(ok_sex))

    # hospital and doctor consistency
    d = claims.merge(doctors, on="doctor_id", how="left", suffixes=("", "_doc"))
    check("every doctor works at the claim's hospital", bool((d["hospital_id"] == d["hospital_id_doc"]).all()))
    d = d.merge(packages[["package_code", "specialty"]], left_on="procedure_package_code",
                right_on="package_code", how="left", suffixes=("", "_pkg"))
    check("doctor specialty matches the package specialty", bool((d["specialty"] == d["specialty_pkg"]).all()))
    h = claims.merge(hospitals, on="hospital_id", how="left").merge(
        packages[["package_code", "specialty"]], left_on="procedure_package_code", right_on="package_code", how="left")
    offers = [(sp in specs.split("|")) for sp, specs in zip(h["specialty"], h["specialties"])]
    check("hospital offers the package specialty", all(offers))

    # summary
    print("\n--- Summary ---")
    print(f"claims: {len(claims):,}   patients: {claims['patient_id'].nunique():,}   "
          f"hospitals used: {claims['hospital_id'].nunique():,}   doctors used: {claims['doctor_id'].nunique():,}")
    print(f"claims per patient (mean): {len(claims) / claims['patient_id'].nunique():.2f}")
    print(f"admission range: {adm.min().date()} to {adm.max().date()}")
    print(f"claimed/reference ratio: mean {ratio.mean():.3f}, min {ratio.min():.3f}, max {ratio.max():.3f}")
    print(f"length of stay (mean): {claims['length_of_stay'].mean():.2f} days")
    same = claims.duplicated(subset=["patient_id", "procedure_package_code"], keep=False)
    print(f"claims sharing patient+package with another claim (genuine repeats): {same.mean():.1%}")
    mix = (claims["procedure_package_code"].value_counts(normalize=True)
           .rename_axis("package_code").reset_index(name="share")
           .merge(packages[["package_code", "package_name"]], on="package_code"))
    mix["share"] = (mix["share"] * 100).round(1).astype(str) + "%"
    print("\nPackage mix:")
    print(mix.to_string(index=False))

    print("\nResult:", "ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
