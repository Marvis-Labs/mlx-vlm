import yaml

from ci.change_rules import ChangeContext, ChangeMatch
from ci.kv_cache_change import KVCacheChange

BASE = """
SETTING = 1
class KVCache:
    value = 1
class SimpleKVCache:
    value = 1
class RotatingKVCache:
    value = 1
class BufferedRotatingKVCache:
    value = 1
class ArraysCache:
    value = 1
"""


class FakeSource:
    def __init__(self, base=BASE, head=BASE):
        self.values = {"base": base, "head": head}

    def read_text(self, revision, path):
        return self.values[revision]


def match(path="mlx_vlm/models/cache.py"):
    return ChangeMatch("kv_cache_change", "kv_cache_change", path, {})


def context(*, head="head", target="target"):
    return ChangeContext.create(
        ["mlx_vlm/models/cache.py"],
        head_sha=head,
        base_sha="base",
        target_sha=target,
    )


def component(tmp_path, head=BASE):
    config = tmp_path / "profiles.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "source": "mlx_vlm/models/cache.py",
                "profiles": {
                    "dense": {
                        "implementations": ["KVCache", "SimpleKVCache"],
                        "contract": "ci.kv_cache_profiles.dense:dense_contract_cases",
                    },
                    "windowed": {
                        "implementations": [
                            "RotatingKVCache",
                            "BufferedRotatingKVCache",
                        ],
                        "contract": "ci.kv_cache_profiles.windowed:windowed_contract_cases",
                    },
                    "recurrent": {
                        "implementations": ["ArraysCache"],
                        "contract": None,
                    },
                },
            }
        )
    )
    return KVCacheChange(config, FakeSource(head=head))


def changed(*replacements):
    value = BASE
    for old, new in replacements:
        value = value.replace(old, new)
    return value


def test_cache_change_emits_only_the_touched_profile(tmp_path):
    planner = component(
        tmp_path,
        changed(("class KVCache:\n    value = 1", "class KVCache:\n    value = 2")),
    )
    plan = planner.plan([match()], context())

    assert plan["component"] == "kv_cache_change"
    assert plan["blocked"] == []
    assert plan["gates"] == []
    assert [job["id"] for job in plan["jobs"]] == ["kv_cache_change:dense"]
    job = plan["jobs"][0]
    assert job["head_sha"] == "head"
    assert job["contract_sha"] == "target"
    assert job["kv_cache_contract"] == {
        "profile": "dense",
        "implementations": ["KVCache", "SimpleKVCache"],
        "oracle": "independent_semantic_contract",
        "entry_point": "ci.kv_cache_profiles.dense:dense_contract_cases",
    }


def test_two_classes_in_one_profile_deduplicate_to_one_job(tmp_path):
    planner = component(
        tmp_path,
        changed(
            ("class KVCache:\n    value = 1", "class KVCache:\n    value = 2"),
            (
                "class SimpleKVCache:\n    value = 1",
                "class SimpleKVCache:\n    value = 2",
            ),
        ),
    )

    plan = planner.plan([match(), match()], context())

    assert [job["id"] for job in plan["jobs"]] == ["kv_cache_change:dense"]
    assert plan["metadata"]["symbols"] == ["KVCache", "SimpleKVCache"]


def test_different_profiles_emit_one_job_per_profile(tmp_path):
    planner = component(
        tmp_path,
        changed(
            ("class KVCache:\n    value = 1", "class KVCache:\n    value = 2"),
            (
                "class RotatingKVCache:\n    value = 1",
                "class RotatingKVCache:\n    value = 2",
            ),
            (
                "class BufferedRotatingKVCache:\n    value = 1",
                "class BufferedRotatingKVCache:\n    value = 2",
            ),
        ),
    )

    jobs = planner.plan([match()], context())["jobs"]

    assert [job["id"] for job in jobs] == [
        "kv_cache_change:dense",
        "kv_cache_change:windowed",
    ]


def test_unimplemented_profile_is_explicitly_blocked(tmp_path):
    planner = component(
        tmp_path,
        changed(
            ("class ArraysCache:\n    value = 1", "class ArraysCache:\n    value = 2")
        ),
    )

    plan = planner.plan([match()], context())

    assert plan["jobs"] == []
    assert plan["blocked"][0]["reason"] == "contract_profile_not_implemented"
    assert plan["blocked"][0]["profile"] == "recurrent"


def test_cache_change_uses_target_commit_not_merge_base(tmp_path):
    planner = component(tmp_path, changed(("value = 1", "value = 2")))
    jobs = planner.plan([match()], context())["jobs"]

    assert all(job["contract_sha"] == "target" for job in jobs)
    assert all(job["contract_sha"] != "base" for job in jobs)


def test_cache_change_blocks_without_both_immutable_revisions(tmp_path):
    plan = component(tmp_path).plan([match()], context(head=None, target=None))

    assert plan["jobs"] == []
    assert plan["blocked"][0]["reason"] == "missing_immutable_revisions"
    assert plan["blocked"][0]["missing"] == ["head_sha", "contract_sha"]
