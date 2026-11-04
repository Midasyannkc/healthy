# Data source: what this is and what it is not

## The files

`data/cms/cms_bene`, `cms_carrier`, `cms_inpatient`, `cms_outpatient` and `cms_pde`, each as parquet and csv.

| File | Rows | Content |
| --- | --- | --- |
| bene | 998 | 500 beneficiaries, one row per year for 2008 and 2009: birth date, enrollment months, 11 chronic condition flags (including diabetes), annual Medicare payment totals |
| carrier | 16,677 | Physician and supplier claims with ICD-9 diagnosis codes |
| inpatient | 225 | Hospital stays |
| outpatient | 2,827 | Outpatient facility claims |
| pde | 17,443 | Prescription drug fills (NDC product id, days supply, cost) |

## Where it comes from

The CMS 2008-2010 Data Entrepreneurs' Synthetic Public Use File (DE-SynPUF), published by the Centers for Medicare & Medicaid Services: https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf

These 500 beneficiaries are the sample redistributed by the open source R package `claimsdb` (https://github.com/jfangmeier/claimsdb, MIT license), drawn from CMS DE-SynPUF Sample 2. `scripts/export_cms.py` converts them to parquet and csv.

## Read this before using the results

* **Synthetic, not real patients.** CMS built DE-SynPUF from real Medicare claims and then altered it to prevent re-identification. CMS states that the process changes how variables vary together, so findings should not be taken as facts about the real Medicare population. CMS does not publish person-level real claims openly.
* **Not the full file.** cms.gov downloads are blocked from the build environment, so the full 2008 to 2010 samples (hundreds of thousands of beneficiaries) could not be fetched. Only 500 people are used, which makes every figure noisy. To scale up, download a sample from the CMS page above and point `scripts/export_cms.py` at it.
* **Medicare is not underwritten.** Beneficiaries are mostly 65 and older and are covered regardless of health. The data is a stand-in for an applicant pool, and an insurer's real applicants would be younger and different.
* **The label is a claims flag.** SP_DIABETES is CMS's chronic condition indicator derived from claims, so it means "diagnosed and billed", not "has the disease". People who use more care are more likely to be flagged.
* **Weight and drug class are not in the data.** Claims carry no weight or BMI, so obesity and BMI diagnosis codes (ICD-9 278 and V85) stand in. Drug fills carry NDC product ids, and no NDC to drug class lookup was available, so medications enter as counts and costs, not classes such as insulin or metformin.
