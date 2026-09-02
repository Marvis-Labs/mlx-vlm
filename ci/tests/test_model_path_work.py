from pathlib import Path

from ci import model_path_work


def test_legacy_entrypoint_forwards_to_registered_executor(tmp_path, monkeypatch):
    job = tmp_path / "job.json"
    base = tmp_path / "base"
    head = tmp_path / "head"
    image = tmp_path / "image.jpg"
    received = []

    def fake_main(arguments):
        received.extend(arguments)
        return 7

    monkeypatch.setattr("ci.work_executor.main", fake_main)

    assert (
        model_path_work.main(
            [
                "--job",
                str(job),
                "--profiles",
                str(tmp_path / "profiles.yaml"),
                "--base",
                str(base),
                "--head",
                str(head),
                "--synthetic-compare",
                str(tmp_path / "synthetic.py"),
                "--hf-compare",
                str(tmp_path / "hf.py"),
                "--image",
                str(image),
                "--max-tokens",
                "32",
            ]
        )
        == 7
    )
    control = Path(model_path_work.__file__).resolve().parent.parent
    assert received == [
        "--job",
        str(job),
        "--control",
        str(control),
        "--base",
        str(base),
        "--head",
        str(head),
        "--max-tokens",
        "32",
        "--image",
        str(image),
    ]
