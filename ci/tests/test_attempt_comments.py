import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_attempt_script_posts_every_result_as_a_new_comment(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" >> "$GH_CALLS"\n'
        'case "$*" in\n'
        "  *'?per_page=100'*) [ \"${GH_EXISTING:-0}\" = 1 ] && printf '99\\n'; exit 0 ;;\n"
        "esac\n"
        'cat >> "$GH_INPUTS"\n'
    )
    fake_gh.chmod(0o755)
    summary = tmp_path / "summary.md"
    summary.write_text(
        "<!-- mlx-vlm:ci:attempt:123 -->\nCommit: `abc123`\nStatus: **Passed**\n"
    )
    calls = tmp_path / "calls"
    inputs = tmp_path / "inputs"
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "GH_CALLS": str(calls),
        "GH_INPUTS": str(inputs),
        "CI_REPOSITORY": "org/repo",
        "CI_PR": "7",
        "CI_SUMMARY": str(summary),
        "CI_ATTEMPT_ID": "123",
    }

    script = ROOT / ".github/scripts/post-ci-attempt.sh"
    subprocess.run(["bash", script], check=True, env=env)
    summary.write_text(
        "<!-- mlx-vlm:ci:attempt:124 -->\nCommit: `def456`\nStatus: **Passed**\n"
    )
    env["CI_ATTEMPT_ID"] = "124"
    subprocess.run(["bash", script], check=True, env=env)

    posted = [
        call for call in calls.read_text().splitlines() if "--method POST" in call
    ]
    assert len(posted) == 2
    assert all(
        "--method POST repos/org/repo/issues/7/comments" in call for call in posted
    )


def test_attempt_script_does_not_duplicate_an_existing_attempt(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        "#!/bin/sh\nprintf '99\\n'\nprintf '%s\\n' \"$*\" >> \"$GH_CALLS\"\n"
    )
    fake_gh.chmod(0o755)
    summary = tmp_path / "summary.md"
    summary.write_text(
        "<!-- mlx-vlm:ci:attempt:123 -->\nCommit: `abc123`\nStatus: **Passed**\n"
    )
    calls = tmp_path / "calls"
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "GH_CALLS": str(calls),
        "CI_REPOSITORY": "org/repo",
        "CI_PR": "7",
        "CI_SUMMARY": str(summary),
        "CI_ATTEMPT_ID": "123",
    }

    script = ROOT / ".github/scripts/post-ci-attempt.sh"
    subprocess.run(["bash", script], check=True, env=env)

    assert "--method POST" not in calls.read_text()


def test_plan_and_attempt_comments_have_independent_markers():
    upsert = (ROOT / ".github/scripts/upsert-ci-plan.sh").read_text()
    workflow = (ROOT / ".github/workflows/bench.yml").read_text()

    assert "<!-- mlx-vlm:ci:plan -->" in upsert
    assert "post-ci-attempt.sh" in workflow
    assert "cancel-in-progress" not in workflow


def test_benchmark_workflow_uses_atomic_device_leases():
    workflow = (ROOT / ".github/workflows/bench.yml").read_text()

    assert "contents: write" in workflow
    assert "ci.device_lease acquire" in workflow
    assert "ci.device_lease heartbeat-loop" in workflow
    assert "ci.device_lease release" in workflow
    assert "lease.json" in workflow
    assert '--target-sha "${{ github.workflow_sha }}"' in workflow
