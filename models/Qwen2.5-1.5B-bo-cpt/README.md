---
license: apache-2.0
datasets:
- pkupie/mc2_corpus
language:
- bo
base_model:
- Qwen/Qwen2.5-1.5B
pipeline_tag: text-generation
---
# Qwen2.5-1.5B Continually Pretrained on Tibetan

This model is a continual pretraining (CPT) checkpoint built by further pretraining Qwen2.5 1.5B on the Tibetan portion of the [MC^2 Corpus](https://huggingface.co/datasets/pkupie/mc2_corpus).

The model is intended to improve Tibetan language modeling and to support research on low-resource language adaptation.

Training details and methodology are described in: ["Efficient Low-Resource Language Adaptation via Multi-Source Dynamic Logit Fusion"](https://arxiv.org/abs/2604.18106) (ACL 2026).

## Training Data

* **Corpus:** Tibetan subset of MC^2 Corpus
* **Language:** Tibetan (`bo`)
* **Training paradigm:** Continual pretraining (CPT) starting from Qwen2.5-1.5B


## Intended Use

This checkpoint is released primarily for research purposes. Researchers are welcome to use this CPT checkpoint as a base model for future work, particularly in model merging and logit fusion.


## Citation

If you use this model, please cite:

```bibtex
@article{zhang2026efficient,
  title={Efficient Low-Resource Language Adaptation via Multi-Source Dynamic Logit Fusion},
  author={Zhang, Chen and Lin, Jiuheng and Liao, Zhiyuan and Feng, Yansong},
  journal={arXiv preprint arXiv:2604.18106},
  year={2026}
}
```
