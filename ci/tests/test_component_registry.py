from ci.components import registry


def test_component_registrations_are_unique_and_self_contained():
    names = [registration.name for registration in registry.REGISTRATIONS]

    assert len(names) == len(set(names))
    assert registry.supported_work() == frozenset(
        {
            ("ModelPath", "model_path"),
            ("KVCacheChange", "kv_cache_change"),
        }
    )
    assert registry.supported_phases() == frozenset(
        {"mlp_contract", "kv_cache_contract", "synthetic", "hf_checkpoint"}
    )


def test_removing_a_registration_removes_its_work_and_phases(monkeypatch):
    retained = tuple(
        registration
        for registration in registry.REGISTRATIONS
        if registration.name != "kv_cache_change"
    )
    monkeypatch.setattr(registry, "REGISTRATIONS", retained)

    assert ("KVCacheChange", "kv_cache_change") not in registry.supported_work()
    assert "kv_cache_contract" not in registry.supported_phases()
