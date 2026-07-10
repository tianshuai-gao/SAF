Test-set metric files, one tree per method.
Layout: {method}/{model_family}/{lang}/{task}/.
Filenames carry the full key: {method}_{family}_{lang}_{task}_{scale}.json.
single uses ins{scale} or cpt{size} in the scale slot. saf uses inshost{scale} or cpthost{scale}.
uniform is the delta-zero ablation of SAF-W and decodes math.
All files start as 0-byte placeholders and are filled from CSD3 after restoration.
Each filled json carries a metadata header: method, family, lang, task, scale, metric, value, n, and source_run (the original CSD3 filename).
Check remaining unfilled files with: find results/test_metrics -name "*.json" -size 0 | wc -l
