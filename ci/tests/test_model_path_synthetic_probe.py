from ci.model_path_synthetic_probe import ADAPTERS


def test_stress_model_families_have_registered_synthetic_adapters():
    assert {
        "deepseek_vl_v2",
        "granite_vision",
        "internvl_chat",
        "qwen2_vl",
        "qwen2_5_vl",
    } <= ADAPTERS.keys()
