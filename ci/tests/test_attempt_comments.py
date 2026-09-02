import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
CONTROL_PLANE_SHA = "2d048c7da55054fc58ec28af8f60262bd3800c24"


def test_execution_workflow_is_a_thin_authorized_caller():
    workflow = (ROOT / ".github/workflows/bench.yml").read_text()

    assert "github.event.comment.body == '/ci run'" in workflow
    assert "author_association" not in workflow
    assert "pull_request_target" not in workflow
    assert "secrets: inherit" not in workflow
    assert "ci.device_lease" not in workflow
    assert "RUN_JOB.sh" not in workflow
    assert CONTROL_PLANE_SHA in workflow


def test_plan_workflow_is_a_thin_control_plane_caller():
    workflow = (ROOT / ".github/workflows/ci-control.yml").read_text()

    assert "pull_request:" in workflow
    assert "pull_request_target" not in workflow
    assert "ci.control" not in workflow
    assert "ci.hosted_checks" not in workflow
    assert CONTROL_PLANE_SHA in workflow


def test_workflows_pin_actions_and_minimize_default_permissions():
    workflows = ROOT / ".github/workflows"

    for path in workflows.glob("*.yml"):
        source = path.read_text()
        assert "permissions: {}" in source
        assert re.search(r"^\s*pull_request_target:", source, re.MULTILINE) is None
        assert re.search(r"^\s*workflow_run:", source, re.MULTILINE) is None
        for line in source.splitlines():
            if re.search(r"^\s*-?\s*uses:", line):
                revision = line.split("@", 1)[1].split()[0]
                assert re.fullmatch(r"[0-9a-f]{40}", revision)
