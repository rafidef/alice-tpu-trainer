#!/usr/bin/env python3
"""Regression tests for Plan B model catch-up hotfixes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
MINER_DIR = REPO_ROOT / "miner"
if str(MINER_DIR) not in sys.path:
    sys.path.insert(0, str(MINER_DIR))

import plan_b


class PlanBHotfixTests(unittest.TestCase):
    def test_apply_epoch_updates_redownloads_full_model_when_local_is_ahead_of_published_updates(self) -> None:
        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
        trainer.model = object()
        trainer.current_model_version = 140
        trainer._publication_state = mock.Mock(
            return_value={
                "target_version": 149,
                "bootstrap_version": 149,
                "published_update_version": 136,
                "full_model_base_urls": ["https://dl.aliceprotocol.org/models"],
                "epoch_update_base_urls": ["https://dl.aliceprotocol.org/epoch_updates"],
            }
        )
        trainer._select_full_download_version = mock.Mock(return_value=149)
        trainer.download_full_model = mock.Mock()
        trainer._download_epoch_update_from_mirrors = mock.Mock()
        trainer._download_epoch_update_from_ps = mock.Mock()

        trainer.apply_epoch_updates()

        trainer._select_full_download_version.assert_called_once_with(149, 149)
        trainer.download_full_model.assert_called_once_with(
            149,
            mirror_urls=["https://dl.aliceprotocol.org/models"],
        )
        trainer._download_epoch_update_from_mirrors.assert_not_called()
        trainer._download_epoch_update_from_ps.assert_not_called()

    def test_wait_for_next_epoch_skips_wait_when_local_matches_live_full_model(self) -> None:
        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
        trainer.current_model_version = 150
        trainer._publication_state = mock.Mock(
            return_value={
                "target_version": 150,
                "published_full_version": 150,
                "published_update_version": 140,
            }
        )

        with mock.patch.object(plan_b.time, "sleep") as sleep_mock:
            plan_b.wait_for_next_epoch(trainer, poll_interval_s=1)

        sleep_mock.assert_not_called()

    def test_wait_for_next_epoch_skips_wait_when_no_newer_published_artifact_exists(self) -> None:
        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
        trainer.current_model_version = 150
        trainer._publication_state = mock.Mock(
            return_value={
                "target_version": 149,
                "published_full_version": 149,
                "published_update_version": 140,
            }
        )

        with mock.patch.object(plan_b.time, "sleep") as sleep_mock:
            plan_b.wait_for_next_epoch(trainer, poll_interval_s=1)

        sleep_mock.assert_not_called()

    def test_wait_for_next_epoch_still_waits_for_newer_published_version(self) -> None:
        trainer = plan_b.LocalTrainer.__new__(plan_b.LocalTrainer)
        trainer.current_model_version = 149
        trainer._publication_state = mock.Mock(
            return_value={
                "target_version": 150,
                "published_full_version": 150,
                "published_update_version": 140,
            }
        )

        with mock.patch.object(plan_b.time, "sleep") as sleep_mock:
            plan_b.wait_for_next_epoch(trainer, poll_interval_s=1)

        sleep_mock.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
