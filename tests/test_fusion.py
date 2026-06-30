"""Unit tests for the fusion rules of SAF-W, Proxy Tuning, and TriMix.

The tests construct each decoder with ``__new__`` to bypass model loading and
exercise ``fuse_logits`` directly, so they need no model weights and run on CPU.
"""

import torch

from safw.dexperts import DExpertsLlama, TriMixQwen, SAFW


def test_proxy_residual():
    """Proxy Tuning: L_base + alpha * (L_expert - L_antiexpert)."""
    d = DExpertsLlama.__new__(DExpertsLlama)
    d.alpha = 1.0
    lb = torch.tensor([[1.0, 0.0, 0.0]])
    le = torch.tensor([[0.0, 2.0, 0.0]])
    la = torch.tensor([[0.0, 0.0, 1.0]])
    fused = d.fuse_logits(lb, le, la)
    assert torch.allclose(fused, lb + (le - la))


def test_trimix_weighted_sum():
    """TriMix: weighted sum, antiexpert weight = 1 - base - expert."""
    t = TriMixQwen.__new__(TriMixQwen)
    t.base_weight = 0.1
    t.expert_weight = 0.9
    t.plausibility_model = "none"
    t._plausibility_alpha_tensor = None
    lb = torch.tensor([[1.0, 0.0, 0.0]])
    le = torch.tensor([[0.0, 1.0, 0.0]])
    la = torch.tensor([[0.0, 0.0, 1.0]])
    fused = t.fuse_logits(lb, le, la)
    assert torch.allclose(fused, 0.1 * lb + 0.9 * le + 0.0 * la)


def test_trimix_plausibility_masks():
    """TriMix plausibility constraint masks implausible tokens to -inf."""
    t = TriMixQwen.__new__(TriMixQwen)
    t.base_weight = 0.1
    t.expert_weight = 0.9
    t.plausibility_model = "expert"
    t.plausibility_alpha = 0.5
    t._plausibility_alpha_tensor = None
    lb = torch.tensor([[5.0, 5.0, 5.0]])
    le = torch.tensor([[10.0, 0.0, 0.0]])
    la = torch.tensor([[0.0, 0.0, 0.0]])
    fused = t.fuse_logits(lb, le, la)
    assert fused[0, 0] > float("-inf")
    assert fused[0, 1] == float("-inf")
    assert fused[0, 2] == float("-inf")


def test_safw_endorse_leans_host():
    """SAF-W: scorer endorses host top token -> small beta -> host wins."""
    s = SAFW.__new__(SAFW)
    s.fixed_beta = False
    l_host = torch.tensor([[4.0, 0.0, 0.0]])
    l_scorer = torch.tensor([[6.0, 0.0, 0.0]])
    fused = s.fuse_logits(l_host, l_scorer, torch.zeros_like(l_host))
    assert int(fused.argmax()) == 0


def test_safw_reject_leans_scorer():
    """SAF-W: scorer rejects host top token -> large beta -> scorer wins."""
    s = SAFW.__new__(SAFW)
    s.fixed_beta = False
    l_host = torch.tensor([[4.0, 0.0, 0.0]])
    l_scorer = torch.tensor([[0.0, 0.0, 7.0]])
    fused = s.fuse_logits(l_host, l_scorer, torch.zeros_like(l_host))
    assert int(fused.argmax()) == 2


def test_safw_fixed_is_uniform_averaging():
    """SAF-W with fixed beta=0.5 reduces to uniform averaging."""
    s = SAFW.__new__(SAFW)
    s.fixed_beta = True
    s.beta_fixed = 0.5
    l_host = torch.tensor([[2.0, 0.0]])
    l_scorer = torch.tensor([[0.0, 4.0]])
    fused = s.fuse_logits(l_host, l_scorer, torch.zeros_like(l_host))
    assert torch.allclose(fused, 0.5 * l_host + 0.5 * l_scorer)
