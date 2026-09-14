import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
APP_ACTION_SHA = "bcd2ba49218906704ab6c1aa796996da409d3eb1"


def test_benchmark_dispatches_only_authorized_identity():
    workflow = (ROOT / ".github/workflows/bench.yml").read_text()

    assert "issue_comment:" in workflow
    assert "permissions: {}" in workflow
    assert f"uses: actions/create-github-app-token@{APP_ACTION_SHA}" in workflow
    assert "repos/Marvis-Labs/mlx-ci/dispatches" in workflow
    assert "github.event.comment.body == '/ci run'" in workflow
    assert "collaborators/$COMMENTER/permission" in workflow
    assert "client_payload" in workflow
    assert "Blaizzy/mlx-vlm" not in workflow
    assert "author_association" not in workflow
    assert "self-hosted" not in workflow
    assert "secrets: inherit" not in workflow
    assert "actions/checkout" not in workflow
    assert "github.event.pull_request.head" not in workflow


def test_pull_request_plan_dispatches_metadata_only():
    workflow = (ROOT / ".github/workflows/ci-control.yml").read_text()

    assert "pull_request_target:" in workflow
    assert "permissions: {}" in workflow
    assert 'event_type:"ci-plan-request"' in workflow
    assert "repos/Marvis-Labs/mlx-ci/dispatches" in workflow
    assert "actions/checkout" not in workflow
    assert "github.event.pull_request.head" not in workflow
    assert "PYTHONPATH" not in workflow
    assert "secrets: inherit" not in workflow


def test_repository_keeps_only_repository_owned_interfaces():
    assert (ROOT / "ci/plugin.py").is_file()
    assert (ROOT / "ci/model_path/planning.py").is_file()
    assert (ROOT / "ci/model_path/synthetic_probe.py").is_file()
    assert (ROOT / "ci/model_path/checkpoint_probe.py").is_file()
    assert (ROOT / "ci/requirements.txt").is_file()
    for name in (
        "control.py",
        "execution_security.py",
        "repository_adapter.py",
        "report.py",
        "work_executor.py",
        "worker_result.py",
    ):
        assert not (ROOT / "ci" / name).exists()


def test_workflows_pin_actions_and_minimize_default_permissions():
    workflows = ROOT / ".github/workflows"

    for name in ("bench.yml", "ci-control.yml", "tests.yml"):
        path = workflows / name
        source = path.read_text()
        assert "permissions: {}" in source
        if name != "ci-control.yml":
            assert re.search(r"^\s*pull_request_target:", source, re.MULTILINE) is None
        assert re.search(r"^\s*workflow_run:", source, re.MULTILINE) is None
        for line in source.splitlines():
            if re.search(r"^\s*-?\s*uses:", line):
                assert re.search(r"uses:\s+[^@\s]+@[0-9a-f]{40}(?:\s|$)", line)
