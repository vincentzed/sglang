"""Policy tests for SigLIP2's platform-specific attention default.

Run with:

    python -m pytest test/registered/models/test_siglip2_attention_backend.py -v
"""

from unittest.mock import patch

from sglang.srt.models.siglip2 import _default_siglip2_qkv_backend
from sglang.test.ci.ci_register import register_amd_ci, register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_amd_ci(est_time=10, stage="stage-a", runner_config="1-gpu-small-amd")


def test_siglip2_defaults_to_triton_on_rocm():
    with patch("sglang.srt.models.siglip2.is_hip", return_value=True):
        assert _default_siglip2_qkv_backend() == "triton_attn"


def test_siglip2_preserves_platform_default_off_rocm():
    with patch("sglang.srt.models.siglip2.is_hip", return_value=False):
        assert _default_siglip2_qkv_backend() is None
