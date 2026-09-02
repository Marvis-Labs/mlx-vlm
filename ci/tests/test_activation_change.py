import yaml

from ci.activation_change import ActivationChange
from ci.change_rules import ChangeContext, ChangeMatch

BASE = """
from functools import partial
VALUE = 1

@partial(object, enabled=True)
def swiglu(gate, value):
    return gate * value

def xielu(value, alpha_p, alpha_n, beta, eps):
    return value

class XieLU:
    def __call__(self, value):
        return value
"""


class FakeSource:
    def __init__(self, head=BASE):
        self.values = {"base": BASE, "head": head}

    def read_text(self, revision, path):
        return self.values[revision]


def component(tmp_path, head=BASE):
    config = tmp_path / "activation.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "source": "mlx_vlm/models/activations.py",
                "profiles": {
                    "swiglu": {
                        "symbols": ["swiglu"],
                        "downstream": ["SwiGLUMLP", "SwitchGLU"],
                    },
                    "xielu": {
                        "symbols": ["xielu", "XieLU"],
                        "downstream": ["apertus"],
                    },
                },
            }
        )
    )
    return ActivationChange(config, FakeSource(head))


def match():
    return ChangeMatch(
        "activation_change",
        "activation_change",
        "mlx_vlm/models/activations.py",
        {},
    )


def context(*, base="base", head="head", target="target"):
    return ChangeContext.create(
        ["mlx_vlm/models/activations.py"],
        base_sha=base,
        head_sha=head,
        target_sha=target,
    )


def changed(old, new):
    return BASE.replace(old, new)


def test_swiglu_change_emits_one_activation_job(tmp_path):
    planner = component(tmp_path, changed("return gate * value", "return value * gate"))

    plan = planner.plan([match()], context())

    assert plan["blocked"] == []
    assert [job["id"] for job in plan["jobs"]] == ["activation_change:swiglu"]
    job = plan["jobs"][0]
    assert job["work_type"] == "ActivationChange"
    assert job["phases"] == ["activation_contract"]
    assert job["activation_contract"] == {
        "profile": "swiglu",
        "symbols": ["swiglu"],
        "downstream": ["SwiGLUMLP", "SwitchGLU"],
        "oracle": "independent_mathematical_contract",
    }


def test_xielu_function_and_module_changes_deduplicate_to_one_job(tmp_path):
    head = changed("return value\n\nclass XieLU", "return value + 1\n\nclass XieLU")
    head = head.replace("return value\n", "return value - 1\n", 1)
    planner = component(tmp_path, head)

    plan = planner.plan([match(), match()], context())

    assert [job["id"] for job in plan["jobs"]] == ["activation_change:xielu"]
    assert plan["metadata"]["symbols"] == ["XieLU", "xielu"]


def test_changes_to_both_profiles_emit_independent_jobs(tmp_path):
    head = changed("return gate * value", "return value * gate")
    head = head.replace(
        "return value\n\nclass XieLU", "return value + 1\n\nclass XieLU"
    )
    planner = component(tmp_path, head)

    jobs = planner.plan([match()], context())["jobs"]

    assert [job["id"] for job in jobs] == [
        "activation_change:swiglu",
        "activation_change:xielu",
    ]


def test_module_level_semantic_change_conservatively_runs_every_profile(tmp_path):
    planner = component(tmp_path, BASE.replace("VALUE = 1", "VALUE = 2"))

    plan = planner.plan([match()], context())

    assert [job["profile"] for job in plan["jobs"]] == ["swiglu", "xielu"]
    assert plan["metadata"]["detection"] == "conservative_module_change"


def test_comment_only_change_emits_no_jobs(tmp_path):
    planner = component(tmp_path, "# formatting only\n" + BASE)

    plan = planner.plan([match()], context())

    assert plan["jobs"] == []
    assert plan["metadata"]["detection"] == "no_semantic_change"


def test_missing_revisions_block_activation_work(tmp_path):
    plan = component(tmp_path).plan(
        [match()], context(base=None, head=None, target=None)
    )

    assert plan["jobs"] == []
    assert plan["blocked"][0]["reason"] == "missing_immutable_revisions"
    assert plan["blocked"][0]["missing"] == [
        "base_sha",
        "head_sha",
        "contract_sha",
    ]
