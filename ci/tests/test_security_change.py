from pathlib import Path

from ci.change_rules import ChangeContext, ChangeMatch
from ci.security_change import SecurityChange, _new_findings


class Source:
    def __init__(self, values):
        self.values = values

    def read_text(self, revision, path):
        return self.values[(revision, path)]


def planner(values):
    return SecurityChange(
        Path(__file__).parents[1] / "components" / "security.yaml",
        Source(values),
    )


def context(path):
    return ChangeContext.create(
        [path],
        base_files=[path],
        head_files=[path],
        base_sha="base",
        head_sha="head",
        target_sha="target",
        tree_state_known=True,
    )


def match(path):
    return ChangeMatch("security_change", "security_change", path, {})


def test_security_change_blocks_new_unsafe_deserialization():
    path = "mlx_vlm/models/example/convert.py"
    values = {
        ("base", path): "def load(path):\n    return path\n",
        (
            "head",
            path,
        ): "def load(path):\n    return torch.load(\n        path, map_location='cpu'\n    )\n",
    }

    plan = planner(values).plan([match(path)], context(path))

    check = next(
        check
        for check in plan["checks"]
        if check["profile"] == "unsafe_deserialization"
    )
    assert check["status"] == "blocked"
    assert plan["blocked"][0]["reason"] == "security_policy_violation"
    assert plan["blocked"][0]["findings"] == [
        {
            "category": "unsafe_deserialization",
            "rule": "torch_load_without_weights_only",
        }
    ]


def test_security_change_allows_removing_unsafe_behavior():
    path = "mlx_vlm/utils.py"
    values = {
        ("base", path): "def load(x):\n    return exec(x)\n",
        ("head", path): "def load(x):\n    return x\n",
    }

    plan = planner(values).plan([match(path)], context(path))

    assert plan["blocked"] == []
    assert all(check["status"] == "passed" for check in plan["checks"])


def test_security_scan_detects_remote_code_and_shell_execution():
    findings = _new_findings(
        "",
        """
def run(path):
    model = load(path, trust_remote_code=True)
    subprocess.run(path, shell=True)
    return model
""",
    )

    assert {finding["rule"] for finding in findings} == {
        "trust_remote_code_true",
        "subprocess_shell_true",
    }


def test_security_change_blocks_command_execution_in_model_source():
    path = "mlx_vlm/models/example/model.py"
    values = {
        ("base", path): "VALUE = 1\n",
        ("head", path): "VALUE = eval('1')\n",
    }

    plan = planner(values).plan([match(path)], context(path))

    check = next(
        check for check in plan["checks"] if check["profile"] == "source_execution"
    )
    assert check["status"] == "blocked"
    assert check["findings"] == [{"category": "command_execution", "rule": "eval"}]
