from ci.model_path_synthetic_probe import ADAPTERS


def test_qwen_model_families_have_registered_synthetic_adapters():
    assert {"qwen2_vl", "qwen2_5_vl"} <= ADAPTERS.keys()
