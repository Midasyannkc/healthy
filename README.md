# Healthy

A diabetes risk model that shows how an insurer can use medical history, prior visits, a weight proxy, medications and age during underwriting to **accept, rate (price) or refer** each applicant. Built on CMS Medicare claims data.

## Honest scope

* The data is CMS's **synthetic** Medicare sample (DE-SynPUF), 500 beneficiaries, not real patient records and not the full file. See `data/SOURCE.md`.
* Medicare beneficiaries are not underwritten and are mostly 65 or older. This is a demonstration of the method on public data, not a pricing model an insurer could use.
* Weight is not in claims, so obesity and BMI diagnosis codes stand in. Drug classes need an NDC lookup that was not available, so medications are counts and costs.
* The label is a claims-based diabetes flag, so the model partly learns who uses care. A person with no claims history looks low risk to it.
* The model uses no race, sex, geographic or birth date feature, and a test enforces it. It never issues a final denial: the top tier is **referred** to a human underwriter for medical evidence.

## The question

Applicants with no diabetes flag in 2008 are scored as of 1 January 2009 using only 2008 history. The outcome is a diabetes flag during 2009. 296 applicants, 67 of them (22.6%) flagged within the year.

## Results (5 times repeated 5-fold cross validation)

| Model | AUC | 95% bootstrap interval | Top decile lift |
| --- | --- | --- | --- |
| Baseline: age and count of chronic conditions | 0.763 | 0.69 to 0.83 | 2.6x |
| Logistic regression, all features | 0.871 | 0.82 to 0.91 | 3.7x |
| Gradient boosted trees | 0.870 | n/a | 3.2x |

The tree model did not beat the simpler logistic model, so logistic regression is used. Removing the "diabetes code seen on a claim" feature leaves AUC at 0.871.

## Underwriting actions

Risk is divided by the base rate (22.6%) to get a relative risk. Price relativity is the expected 2009 Medicare cost for that risk, divided by the average, capped at 2.5. The bands were set in advance, not tuned on outcomes.

| Action | Rule | Applicants | Diabetes within a year | Average price relativity |
| --- | --- | --- | --- | --- |
| Preferred | relative risk below 0.7 | 188 | 7.4% | 0.59 |
| Standard | 0.7 to 1.3 | 29 | 27.6% | 0.98 |
| Rated (accept, extra premium) | 1.3 to 2.0 | 20 | 45.0% | 1.32 |
| Refer (nurse exam or physician statement) | 2.0 or more | 59 | 61.0% | 2.14 |

Strongest signals in the logistic model: number of visit days, outpatient and physician claims, a hypertension diagnosis, drug cost and age. Expected cost is $5,724 for a person who develops diabetes against $975 for one who does not, in this sample.

## Limits that matter

* 296 applicants and 67 events mean wide intervals. Treat the AUC as roughly 0.8 to 0.9, not 0.87.
* High utilization raises the score. An insurer using this would charge people more for seeking care, which regulators would question.
* Price relativities come from Medicare payments, not an insurer's claim costs or loss ratios.
* Not validated on any other population.

## Run

`pip install -r requirements.txt`, then `make run` (writes `out/report.json` and `out/decisions.csv`) and `make test`. Code: `healthy/features.py` (point-in-time features), `healthy/model.py`, `healthy/underwriting.py`, `healthy/run.py`.
