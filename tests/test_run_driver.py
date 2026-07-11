"""Regression tests for the declaration checks in scripts/run.py."""
import pytest

from scripts.run import build_argparser, resolve_models
from scripts.canonical import resolve


def parse(argv):
    return build_argparser().parse_args(argv)


def test_safw_cpthost_stem():
    args = parse(["--method", "safw", "--lang", "bo",
                  "--scale", "32B", "--host", "cpt"])
    models, _ = resolve_models(args)
    assert models[0].endswith("bo-cpt")
    stem = resolve("safw", "bo", "rc", models)
    assert stem.endswith("saf/qwen/bo/rc/saf_qwen_bo_rc_cpthost32B")


def test_uniform_is_symmetric():
    args = parse(["--method", "safw_fixed", "--lang", "kk", "--scale", "14B"])
    models, _ = resolve_models(args)
    stem = resolve("safw_fixed", "kk", "math", models)
    assert stem.endswith("uniform/qwen/kk/math/uniform_qwen_kk_math_14B")


def test_gemma_with_explicit_cpt():
    args = parse(["--method", "safw", "--family", "gemma", "--lang", "bo",
                  "--scale", "12B", "--host", "ins",
                  "--scorer_model", "pkupie/gemma-3-4B-bo-cpt"])
    models, _ = resolve_models(args)
    stem = resolve("safw", "bo", "title", models)
    assert stem.endswith("saf/gemma/bo/title/saf_gemma_bo_title_inshost12B")


@pytest.mark.parametrize("argv,frag", [
    (["--method", "safw", "--lang", "bo", "--scale", "7B"],
     "--host is required"),
    (["--method", "safw_fixed", "--lang", "bo", "--scale", "7B",
      "--host", "ins"],
     "does not apply to safw_fixed"),
    (["--method", "proxy", "--lang", "bo", "--scale", "7B", "--host", "ins"],
     "applies only to safw"),
    (["--method", "safw", "--lang", "bo", "--scale", "7B", "--host", "ins",
      "--scorer_model", "pkupie/Qwen2.5-1.5B-ug-cpt"],
     "--lang bo but cpt model"),
    (["--method", "safw", "--family", "gemma", "--lang", "bo",
      "--scale", "12B", "--host", "ins",
      "--host_model", "Qwen/Qwen2.5-14B-Instruct",
      "--scorer_model", "pkupie/Qwen2.5-1.5B-bo-cpt"],
     "--family gemma but model"),
    (["--method", "safw", "--lang", "bo", "--scale", "32B", "--host", "cpt",
      "--scorer_model", "Qwen/Qwen2.5-14B-Instruct"],
     "--scale 32B but instruct model"),
    (["--method", "trimix", "--lang", "bo", "--scale", "7B",
      "--antiexpert_model", "pkupie/Qwen2.5-1.5B-bo-cpt"],
     "antiexpert slot holds a cpt"),
    (["--method", "safw", "--lang", "bo", "--scale", "7B", "--host", "ins",
      "--base_model", "Qwen/Qwen2.5-7B-Instruct"],
     "do not apply to SAF-W"),
    (["--method", "proxy", "--family", "gemma", "--lang", "bo",
      "--scale", "12B"],
     "no default cpt model for family gemma"),
])
def test_rejections(argv, frag):
    args = parse(argv)
    with pytest.raises(SystemExit) as excinfo:
        resolve_models(args)
    assert frag in str(excinfo.value)
