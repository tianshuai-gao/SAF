Test-set metric files, one tree per method.
Layout: {method}/{model_family}/{lang}/{task}/{scale}.json.
single holds the individual models, with ins_{scale}.json and cpt_{size}.json.
saf holds both deployments, with inshost_{scale}.json and cpthost_{scale}.json.
uniform is the delta-zero ablation of SAF-W and decodes math.
All files start as 0-byte placeholders and are filled from CSD3 after restoration.
Check remaining unfilled files with: find results/test_metrics -name "*.json" -size 0 | wc -l
