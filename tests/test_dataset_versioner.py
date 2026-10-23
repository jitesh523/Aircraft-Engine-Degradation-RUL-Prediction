"""Regression tests for persistent dataset snapshots."""

import datetime
import json

import pandas as pd

from data_loader import DatasetVersioner


def test_versions_created_at_same_time_preserve_both_snapshots(tmp_path, monkeypatch):
    """Rapid snapshots must remain independently loadable after a restart."""

    class FrozenDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 23, 12, 0, 0, tzinfo=tz)

    monkeypatch.setattr(datetime, "datetime", FrozenDateTime)
    versioner = DatasetVersioner(versions_dir=str(tmp_path))
    first = pd.DataFrame({"sensor_1": [1.0, 2.0]})
    second = pd.DataFrame({"sensor_1": [3.0, 4.0]})

    first_id = versioner.create_version(first, "first")
    second_id = versioner.create_version(second, "second")

    assert first_id != second_id
    restored = DatasetVersioner(versions_dir=str(tmp_path))
    assert len(restored.versions) == 2
    assert restored.current_version == second_id
    pd.testing.assert_frame_equal(restored.get_version(first_id), first)
    pd.testing.assert_frame_equal(restored.get_version(second_id), second)
    pd.testing.assert_frame_equal(restored.rollback(first_id), first)
    assert DatasetVersioner(versions_dir=str(tmp_path)).current_version == first_id


def test_legacy_timestamp_version_remains_loadable(tmp_path):
    """Existing registries and timestamp-only snapshot paths remain valid."""
    version_id = "v_20261022_120000"
    data = pd.DataFrame({"sensor_1": [1.0]})
    path = tmp_path / f"{version_id}.parquet"
    data.to_parquet(path, index=False)
    registry = {
        "versions": {version_id: {"path": str(path)}},
        "current_version": version_id,
    }
    (tmp_path / "registry.json").write_text(json.dumps(registry))

    versioner = DatasetVersioner(versions_dir=str(tmp_path))

    assert versioner.current_version == version_id
    pd.testing.assert_frame_equal(versioner.get_version(version_id), data)
