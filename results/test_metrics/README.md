Test-set metric files. Canonical layout: {model_family}/{lang}/{task}/{method}_{scale}.json.
Methods: ins, cpt, proxy, trimix, safw_inshost, safw_cpthost, uniform.
The cpt file carries its own size (1.5B for Qwen, 4B for Gemma) and has no scale variants.
All files start as 0-byte placeholders and are filled from CSD3 after restoration.
Check remaining unfilled files with: find results/test_metrics -name "*.json" -size 0 | wc -l
