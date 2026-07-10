Test-set outputs, one tree per method.
Layout: {method}/{model_family}/{lang}/{task}/{key}_{kind}.json.
The key is {method}_{family}_{lang}_{task}_{scale}. single uses ins{scale} or cpt{size}. saf uses inshost{scale} or cpthost{scale}.
Each run contributes two files. metrics holds the score and preds holds the per-sample outputs.
uniform is the delta-zero ablation of SAF-W and decodes math.
All files start as 0-byte placeholders and are filled from CSD3 after restoration.
Each filled metrics file carries a metadata header with method, family, lang, task, scale, metric, value, n, and source_run.
Check remaining unfilled files with: find results/test_outputs -name "*.json" -size 0 | wc -l
