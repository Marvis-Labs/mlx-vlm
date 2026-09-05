import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
CONTROL_PLANE_SHA = "6ed7a602a7426be8918696261e2fc59d8997ba96"


def test_benchmark_delegates_shared_orchestration():
    workflow = (ROOT / ".github/workflows/bench.yml").read_text()

    assert "issue_comment:" in workflow
    assert "permissions: {}" in workflow
    assert (
        "uses: Marvis-Labs/mlx-ci/.github/workflows/repository-ci.yml@"
        + CONTROL_PLANE_SHA
        in workflow
    )
    assert "runs-on:" not in workflow
    assert "run:" not in workflow
    assert "author_association" not in workflow
    assert "self-hosted" not in workflow
    assert "secrets: inherit" not in workflow


def test_pull_request_plan_delegates_shared_orchestration():
    workflow = (ROOT / ".github/workflows/ci-control.yml").read_text()

    assert "pull_request:" in workflow
    assert "permissions: {}" in workflow
    assert (
        "uses: Marvis-Labs/mlx-ci/.github/workflows/repository-plan.yml@"
        + CONTROL_PLANE_SHA
        in workflow
    )
    assert "runs-on:" not in workflow
    assert "run:" not in workflow
    assert "pull_request_target:" not in workflow
    assert "secrets: inherit" not in workflow


def test_repository_keeps_only_repository_owned_interfaces():
    assert (ROOT / "ci/control.py").is_file()
    assert (ROOT / "ci/repository_adapter.py").is_file()
    assert (ROOT / "ci/work_executor.py").is_file()
    assert (ROOT / "ci/report.py").is_file()
    assert (ROOT / "ci/hosted-requirements.txt").is_file()
    for name in (
        "attempt_lease.py",
        "device_inventory.py",
        "device_lease.py",
        "runner_selection.py",
        "scheduler.py",
        "workflow_report.py",
    ):
        assert not (ROOT / "ci" / name).exists()


def test_workflows_pin_actions_and_minimize_default_permissions():
    workflows = ROOT / ".github/workflows"

    for path in workflows.glob("*.yml"):
        source = path.read_text()
        assert "permissions: {}" in source
        assert re.search(r"^\s*pull_request_target:", source, re.MULTILINE) is None
        assert re.search(r"^\s*workflow_run:", source, re.MULTILINE) is None
        for line in source.splitlines():
            if re.search(r"^\s*-?\s*uses:", line):
                assert re.search(r"uses:\s+[^@\s]+@[0-9a-f]{40}(?:\s|$)", line)
