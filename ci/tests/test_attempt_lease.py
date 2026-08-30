from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from ci.attempt_lease import AttemptLeaseStore
from ci.tests.test_device_lease import NOW, FakeGitHubClient


def acquire(store, attempt_id, head_sha="abcdef1", now=NOW):
    return store.acquire(
        attempt_id=attempt_id,
        pr_number=7,
        head_sha=head_sha,
        target_sha="1234567",
        run_url=f"https://example.com/{attempt_id}",
        ttl_seconds=300,
        now=now,
    )


def test_same_commit_commands_coalesce_to_one_active_attempt():
    client = FakeGitHubClient()

    def claim(index):
        return acquire(AttemptLeaseStore("org/repo", client), str(index))

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(claim, range(100)))

    owners = [lease for lease, acquired in results if acquired]
    coalesced = [lease for lease, acquired in results if not acquired]
    assert len(owners) == 1
    assert len(coalesced) == 99
    assert set(coalesced) == {owners[0]}


def test_completed_attempt_can_be_released_and_rerun():
    client = FakeGitHubClient()
    store = AttemptLeaseStore("org/repo", client)
    first, acquired = acquire(store, "first")
    assert acquired is True
    assert store.release(first) is True

    second, acquired = acquire(store, "second")
    assert acquired is True
    assert second.attempt_id == "second"


def test_new_commit_has_an_independent_attempt_lease():
    client = FakeGitHubClient()
    store = AttemptLeaseStore("org/repo", client)
    first, first_acquired = acquire(store, "first", "abcdef1")
    second, second_acquired = acquire(store, "second", "abcdef2")

    assert first_acquired is True
    assert second_acquired is True
    assert first.head_sha != second.head_sha


def test_expired_attempt_is_replaced_and_old_owner_cannot_release_it():
    client = FakeGitHubClient()
    store = AttemptLeaseStore("org/repo", client)
    first, _ = acquire(store, "first")
    second, acquired = acquire(store, "second", now=NOW + timedelta(minutes=6))

    assert acquired is True
    assert store.release(first) is False
    assert store.get(7, "abcdef1") == second
