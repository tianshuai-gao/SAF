Test-set outputs, one tree per method.
Layout: {method}/{model_family}/{lang}/{task}/{key}_{kind}.json.
The key is {method}_{family}_{lang}_{task}_{scale}. single uses ins{scale} or cpt{size}. saf uses inshost{scale} or cpthost{scale}.
Each run contributes two files. metrics holds the score and preds holds the per-sample outputs.
uniform is the delta-zero ablation of SAF-W and decodes math.
The files are zero-byte placeholders. The runs could not be completed, because the CSD3 cluster was offline from a cooling failure through the submission deadline. The pipeline writes into them in place once the cluster returns.
Check the unfilled files with: find results/test_outputs -name "*.json" -size 0 | wc -l
