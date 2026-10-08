# Health Insurance Fraud Project: Clean Claims Generator (Week 1)

Generates a **clean, fraud-free** synthetic health-insurance claims dataset. This is the
shared data contract for the whole team: the fraud-injection engine, detectors, adaptive
loop and conformal triage all read the table produced here.

> **Status of Synthea:** this generator does **not** use Synthea. It is the fallback plan
> and a fixed format everyone can code against. If a Synthea run works on your laptop, map
> its output to the same 17 columns and everything downstream stays unchanged.

## Quick start

```bash
pip install -r requirements.txt
python src/generate_claims.py --config config/params.json --out data_full   # ~100k claims, seconds
python src/validate_claims.py --data data_full                              # runs 18 checks
```

Small test run: `python src/generate_claims.py --n-patients 5000 --out data`
(`data/` already contains a 5,000-patient sample.)

The seed is fixed in `config/params.json` (`seed: 42`); the same seed gives a byte-identical file.

## Files

| File | Purpose |
|---|---|
| `config/params.json` | All settings, the 16 treatment packages and their rates, age patterns, weights |
| `src/generate_claims.py` | Builds patients, hospitals, doctors, claims |
| `src/validate_claims.py` | Consistency checks and a summary report |
| `data/claims_clean.csv` | The shared claims table (17 columns) |
| `data/patients.csv`, `hospitals.csv`, `doctors.csv`, `packages.csv` | Reference tables for the fraud engine |

## The claims table (17 columns, in this order)

`claim_id, patient_id, hospital_id, doctor_id, admission_date, discharge_date, patient_age,
patient_sex, patient_district, diagnosis_code, procedure_package_code, claimed_amount,
reference_package_rate, length_of_stay, is_fraud, fraud_type, severity_tier`

The last three are the **answer key**: all `0`, `none`, `none` in the clean file. The
fraud-injection engine will set them. **Never feed them to a model as inputs.**

## What the generator does

1. Builds 300 hospitals across 40 placeholder districts; each offers a set of specialties
   (General Medicine everywhere, larger hospitals offer more).
2. Builds about 1,500 doctors, each attached to one hospital and one specialty.
3. Builds patients with an age and sex mix.
4. Gives each patient `1 + Poisson(0.8)` admissions. The treatment package is chosen from
   prevalence weight x age pattern x sex rule (obstetrics = female, prostate = male).
5. Picks a hospital (80% in the patient's district if one offers the specialty) and a
   doctor with the matching specialty at that hospital.
6. Draws length of stay and a claimed amount close to the reference package rate
   (ratio roughly 0.85 to 1.15).
7. Adds genuine repeat claims for cyclical treatments (dialysis sessions, chemotherapy
   cycles) with the same hospital and doctor. These matter: **fraud injection for
   duplicate claims must look different from these genuine repeats.**

## Known limits (read before using the numbers)

- **Every rate, weight and age pattern is an illustrative placeholder.** Package rates are
  NOT real PM-JAY rates, and package codes (`PKG-01` ...) are internal labels. Replace them
  with published NHA package rates and recalibrate against IRDAI / ICMR-INDIAB statistics
  before reporting results.
- **Package mix does not equal the weights.** The mix is also shaped by the patient age
  mix (for example deliveries come out near 9% of claims). Recalibration is a later step.
- **Districts are placeholders** (`District_001` ...), not real districts.
- **Dialysis and chemotherapy repeats are simplified** (a handful of sessions, not the
  true frequency).
- About 30% of claims share a patient and package with another claim (genuine repeats,
  mostly with long gaps, except dialysis and chemotherapy). Keep this in mind when you
  design duplicate-claim detection features.
- Patients here are people who made at least one claim, so claims per patient (about 2)
  is not a population hospitalisation rate.

## Next steps

1. Fraud-injection engine: upcoding and duplicate claims first, then phantom billing,
   with obvious and subtle tiers and fraud rates of 2%, 5% and 8%.
2. Train / calibration / test split (keep the calibration set separate for conformal prediction).
3. Features and baseline detectors.
