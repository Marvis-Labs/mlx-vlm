from ci.change_rules import ChangeContext, ChangeMatch
from ci.kv_cache_change import KVCacheChange


def match(path="mlx_vlm/models/cache.py"):
    return ChangeMatch("kv_cache_change", "kv_cache_change", path, {})


def context(*, head="head", target="target"):
    return ChangeContext.create(
        ["mlx_vlm/models/cache.py"],
        head_sha=head,
        base_sha="merge-base",
        target_sha=target,
    )


def test_cache_change_emits_one_independent_job_per_implemented_profile():
    plan = KVCacheChange().plan([match()], context())

    assert plan["component"] == "kv_cache_change"
    assert plan["blocked"] == []
    assert plan["gates"] == []
    assert len(plan["jobs"]) == 1
    job = plan["jobs"][0]
    assert job == {
        "id": "kv_cache_change:dense",
        "work_type": "KVCacheChange",
        "component": "kv_cache_change",
        "profile": "dense",
        "changed_paths": ["mlx_vlm/models/cache.py"],
        "phases": ["kv_cache_contract"],
        "minimum_memory_gib": 8,
        "required_disk_gib": 2,
        "head_sha": "head",
        "contract_sha": "target",
        "kv_cache_contract": {
            "profile": "dense",
            "implementations": ["KVCache", "SimpleKVCache"],
            "oracle": "independent_semantic_contract",
        },
    }


def test_cache_change_uses_target_commit_not_merge_base_for_trusted_contract():
    job = KVCacheChange().plan([match()], context())["jobs"][0]

    assert job["contract_sha"] == "target"
    assert job["contract_sha"] != "merge-base"


def test_cache_change_blocks_without_both_immutable_revisions():
    plan = KVCacheChange().plan([match()], context(head=None, target=None))

    assert plan["jobs"] == []
    assert plan["blocked"][0]["reason"] == "missing_immutable_revisions"
    assert plan["blocked"][0]["missing"] == ["head_sha", "contract_sha"]


def test_profile_expansion_produces_independently_schedulable_jobs():
    component = KVCacheChange()
    component.profiles = {
        "dense": ("KVCache",),
        "windowed": ("RotatingKVCache",),
        "segmented": ("ChunkedKVCache",),
    }

    jobs = component.plan([match(), match()], context())["jobs"]

    assert [job["id"] for job in jobs] == [
        "kv_cache_change:dense",
        "kv_cache_change:segmented",
        "kv_cache_change:windowed",
    ]
    assert all(job["minimum_memory_gib"] == 8 for job in jobs)
