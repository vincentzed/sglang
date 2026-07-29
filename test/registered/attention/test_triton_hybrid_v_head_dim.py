from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.layers.attention.triton_backend import _resolve_v_head_dims
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


def test_hybrid_pool_does_not_assume_global_layer_zero_is_full_attention():
    token_to_kv_pool = SimpleNamespace(
        get_v_head_dim=Mock(return_value=128),
        get_value_buffer=Mock(side_effect=AssertionError("must not read layer 0")),
    )
    model_runner = SimpleNamespace(
        model_config=SimpleNamespace(v_head_dim=128, swa_v_head_dim=128),
        token_to_kv_pool=token_to_kv_pool,
    )

    with patch(
        "sglang.srt.layers.attention.triton_backend.mambaish_config",
        return_value=object(),
    ):
        assert _resolve_v_head_dims(model_runner, None) == (128, None)

    token_to_kv_pool.get_v_head_dim.assert_called_once_with()
    token_to_kv_pool.get_value_buffer.assert_not_called()


def test_dense_pool_preserves_layer_zero_lookup():
    value_buffer = SimpleNamespace(shape=(1, 1, 64))
    token_to_kv_pool = SimpleNamespace(get_value_buffer=Mock(return_value=value_buffer))
    model_runner = SimpleNamespace(
        model_config=SimpleNamespace(v_head_dim=64, swa_v_head_dim=64),
        token_to_kv_pool=token_to_kv_pool,
    )

    with patch(
        "sglang.srt.layers.attention.triton_backend.mambaish_config",
        return_value=None,
    ):
        assert _resolve_v_head_dims(model_runner, None) == (64, None)

    token_to_kv_pool.get_value_buffer.assert_called_once_with(0)
