"""CPU-only tests for per-Gaussian lifecycle state (no GPU is ever touched).

All tensors live on CPU and the CUDA-only relocation kernel is mocked. Do NOT
mutate CUDA_VISIBLE_DEVICES at module level - unittest discovery imports this
file into the shared process and the hidden device would break the GPU
contract tests that run after it.
"""

from __future__ import annotations

import unittest
from unittest import mock

try:
    import torch
except ImportError:  # pragma: no cover - optional training dependency
    torch = None

if torch is not None:
    try:
        from cloudstudio_3dgs.training.error_weighted_mcmc import (
            ErrorScoreConfig,
            ErrorScoreState,
            ErrorWeightedMCMCStrategy,
            sample_add_weighted,
        )
        from cloudstudio_3dgs.training.gaussian_lifecycle import (
            FIELD_NAMES,
            GaussianLifecycleState,
        )

        _IMPORT_ERROR = None
    except Exception as exc:  # pragma: no cover - e.g. CUDA-only gsplat build
        ErrorScoreConfig = ErrorScoreState = ErrorWeightedMCMCStrategy = None
        GaussianLifecycleState = None
        sample_add_weighted = None
        FIELD_NAMES = ()
        _IMPORT_ERROR = exc
else:  # pragma: no cover
    _IMPORT_ERROR = ImportError("torch is not installed")

_MODULE = "cloudstudio_3dgs.training.error_weighted_mcmc"


def _requires_module(test_case: unittest.TestCase) -> None:
    if _IMPORT_ERROR is not None:
        test_case.skipTest(f"gaussian_lifecycle unavailable on this host: {_IMPORT_ERROR}")


def _fake_compute_relocation(*, opacities, scales, ratios, binoms):
    """CPU stand-in for the CUDA kernel: identity on opacity and scale.

    The lifecycle tests only care about index bookkeeping, so the returned
    values just have to be finite and correctly shaped.
    """
    return opacities.clone(), scales.clone()


def _make_optimizers(params, *, warm: bool = False) -> dict:
    """One Adam per parameter, as GsplatBackend builds them.

    ``warm=True`` runs a real step first so exp_avg/exp_avg_sq exist and are
    non-zero; a cold Adam has empty state, which would hide reordering bugs.
    """
    optimizers = {
        name: torch.optim.Adam([{"params": [p], "name": name}], lr=1e-3)
        for name, p in params.items()
    }
    if warm:
        loss = sum(p.square().sum() for p in params.values())
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return optimizers


class LifecycleDefaultsTests(unittest.TestCase):
    def setUp(self) -> None:
        _requires_module(self)

    def test_fresh_state_has_documented_defaults_and_dtypes(self) -> None:
        state = GaussianLifecycleState(4)
        self.assertEqual(len(state), 4)
        self.assertTrue(torch.equal(state.error_ema, torch.ones(4)))
        self.assertTrue(torch.equal(state.visibility_ema, torch.zeros(4)))
        self.assertTrue(torch.equal(state.contribution_ema, torch.zeros(4)))
        self.assertTrue(torch.equal(state.anchor_index, torch.full((4,), -1)))
        self.assertTrue(torch.equal(state.anchor_confidence, torch.zeros(4)))
        self.assertTrue(torch.equal(state.generation, torch.zeros(4, dtype=torch.int32)))
        self.assertTrue(torch.equal(state.age, torch.zeros(4, dtype=torch.int32)))
        self.assertTrue(torch.equal(state.parent_id, torch.arange(4)))
        self.assertTrue(torch.equal(state.birth_step, torch.zeros(4, dtype=torch.int32)))
        expected_dtypes = {
            "error_ema": torch.float32,
            "visibility_ema": torch.float32,
            "contribution_ema": torch.float32,
            "anchor_index": torch.int64,
            "anchor_confidence": torch.float32,
            "generation": torch.int32,
            "age": torch.int32,
            "parent_id": torch.int64,
            "birth_step": torch.int32,
        }
        self.assertEqual(set(FIELD_NAMES), set(expected_dtypes))
        for name, dtype in expected_dtypes.items():
            self.assertEqual(getattr(state, name).dtype, dtype, name)

    def test_all_fields_share_the_gaussian_count(self) -> None:
        state = GaussianLifecycleState(7)
        for name in FIELD_NAMES:
            self.assertEqual(int(getattr(state, name).shape[0]), 7, name)
        self.assertEqual(GaussianLifecycleState(0).error_ema.numel(), 0)
        with self.assertRaises(ValueError):
            GaussianLifecycleState(-1)

    def test_on_step_ticks_ages_and_records_the_clock(self) -> None:
        state = GaussianLifecycleState(3)
        state.on_step(5)
        state.on_step(6)
        self.assertTrue(torch.equal(state.age, torch.full((3,), 2, dtype=torch.int32)))
        self.assertEqual(state.current_step, 6)
        state.on_step(7, indices=torch.tensor([0, 2]))
        self.assertTrue(
            torch.equal(state.age, torch.tensor([3, 2, 3], dtype=torch.int32))
        )


class LifecycleGrowTests(unittest.TestCase):
    def setUp(self) -> None:
        _requires_module(self)

    def _seeded(self) -> "GaussianLifecycleState":
        state = GaussianLifecycleState(4)
        state.error_ema = torch.tensor([0.11, 0.22, 0.33, 0.44])
        state.anchor_index = torch.tensor([9, -1, 5, 5])
        state.anchor_confidence = torch.tensor([0.9, 0.0, 0.4, 0.4])
        state.generation = torch.tensor([0, 2, 1, 0], dtype=torch.int32)
        state.age = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
        return state

    def test_existing_error_ema_is_bit_identical_after_growth(self) -> None:
        # THE core WP-1 regression: densification used to reset every score to
        # 1.0 (ErrorScoreState.resize), throwing away the accumulated EMA.
        state = self._seeded()
        before = state.error_ema.clone()
        state.on_grow(torch.tensor([1, 3, 1]), step=700)
        self.assertEqual(len(state), 7)
        self.assertTrue(torch.equal(state.error_ema[:4], before))
        # Every other column of the survivors is untouched as well.
        self.assertTrue(torch.equal(state.anchor_index[:4], torch.tensor([9, -1, 5, 5])))
        self.assertTrue(
            torch.equal(state.age[:4], torch.tensor([10, 20, 30, 40], dtype=torch.int32))
        )

    def test_children_inherit_parent_identity(self) -> None:
        state = self._seeded()
        parents = torch.tensor([1, 3, 1])
        state.on_grow(parents, step=700)
        self.assertTrue(
            torch.allclose(state.error_ema[4:], torch.tensor([0.22, 0.44, 0.22]))
        )
        self.assertTrue(torch.equal(state.anchor_index[4:], torch.tensor([-1, 5, -1])))
        self.assertTrue(
            torch.allclose(state.anchor_confidence[4:], torch.tensor([0.0, 0.4, 0.0]))
        )
        self.assertTrue(
            torch.equal(state.generation[4:], torch.tensor([3, 1, 3], dtype=torch.int32))
        )
        self.assertTrue(torch.equal(state.parent_id[4:], parents))
        self.assertTrue(
            torch.equal(state.birth_step[4:], torch.full((3,), 700, dtype=torch.int32))
        )
        self.assertTrue(torch.equal(state.age[4:], torch.zeros(3, dtype=torch.int32)))
        self.assertTrue(torch.equal(state.visibility_ema[4:], torch.zeros(3)))
        self.assertTrue(torch.equal(state.contribution_ema[4:], torch.zeros(3)))

    def test_birth_factor_scales_only_the_children(self) -> None:
        state = self._seeded()
        before = state.error_ema.clone()
        state.on_grow(torch.tensor([0, 2]), step=10, birth_factor=0.5)
        self.assertTrue(torch.equal(state.error_ema[:4], before))
        self.assertTrue(
            torch.allclose(state.error_ema[4:], torch.tensor([0.055, 0.165]))
        )
        with self.assertRaises(ValueError):
            state.on_grow(torch.tensor([0]), step=11, birth_factor=-1.0)

    def test_grow_rejects_out_of_range_parents_and_accepts_empty(self) -> None:
        state = self._seeded()
        with self.assertRaises(ValueError):
            state.on_grow(torch.tensor([4]), step=1)
        added = state.on_grow(torch.tensor([], dtype=torch.int64), step=3)
        self.assertEqual(len(state), 4)
        self.assertEqual(int(added.numel()), 0)
        self.assertEqual(state.current_step, 3)


class LifecycleRelocateTests(unittest.TestCase):
    def setUp(self) -> None:
        _requires_module(self)

    def test_dead_slot_takes_over_the_source_state(self) -> None:
        state = GaussianLifecycleState(5)
        state.error_ema = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        state.anchor_index = torch.tensor([1, 2, 3, 4, 5])
        state.anchor_confidence = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        state.generation = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        state.age = torch.tensor([5, 6, 7, 8, 9], dtype=torch.int32)
        state.visibility_ema = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])

        dead = torch.tensor([0, 3])
        source = torch.tensor([4, 1])
        state.on_relocate(dead, source, step=1200)

        # Dead slots now describe copies of their sources, NOT the corpses.
        self.assertTrue(
            torch.allclose(state.error_ema, torch.tensor([0.5, 0.2, 0.3, 0.2, 0.5]))
        )
        self.assertTrue(torch.equal(state.anchor_index, torch.tensor([5, 2, 3, 2, 5])))
        self.assertTrue(
            torch.allclose(
                state.anchor_confidence, torch.tensor([0.5, 0.2, 0.3, 0.2, 0.5])
            )
        )
        self.assertTrue(
            torch.equal(
                state.generation, torch.tensor([5, 1, 2, 2, 4], dtype=torch.int32)
            )
        )
        self.assertTrue(torch.equal(state.parent_id, torch.tensor([4, 1, 2, 1, 4])))
        self.assertTrue(
            torch.equal(
                state.birth_step,
                torch.tensor([1200, 0, 0, 1200, 0], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(state.age, torch.tensor([0, 6, 7, 0, 9], dtype=torch.int32))
        )
        self.assertTrue(
            torch.allclose(
                state.visibility_ema, torch.tensor([0.0, 0.5, 0.5, 0.0, 0.5])
            )
        )
        self.assertEqual(len(state), 5)

    def test_source_rows_survive_untouched(self) -> None:
        state = GaussianLifecycleState(4)
        state.error_ema = torch.tensor([0.9, 0.8, 0.7, 0.6])
        state.generation = torch.tensor([3, 3, 3, 3], dtype=torch.int32)
        state.age = torch.tensor([7, 7, 7, 7], dtype=torch.int32)
        state.on_relocate(torch.tensor([0]), torch.tensor([2]), step=99)
        self.assertAlmostEqual(float(state.error_ema[2]), 0.7, places=6)
        self.assertEqual(int(state.generation[2]), 3)
        self.assertEqual(int(state.age[2]), 7)
        self.assertEqual(int(state.parent_id[2]), 2)

    def test_length_mismatch_and_out_of_range_are_rejected(self) -> None:
        state = GaussianLifecycleState(3)
        with self.assertRaises(ValueError):
            state.on_relocate(torch.tensor([0, 1]), torch.tensor([2]), step=1)
        with self.assertRaises(ValueError):
            state.on_relocate(torch.tensor([0]), torch.tensor([3]), step=1)


class LifecyclePruneReindexTests(unittest.TestCase):
    def setUp(self) -> None:
        _requires_module(self)

    def _labelled(self, n: int) -> "GaussianLifecycleState":
        state = GaussianLifecycleState(n)
        # A distinct, order-revealing value per row in every column.
        state.error_ema = torch.arange(n, dtype=torch.float32) / 10.0
        state.anchor_index = torch.arange(n, dtype=torch.int64) * 100
        state.anchor_confidence = torch.arange(n, dtype=torch.float32) / 100.0
        state.generation = torch.arange(n, dtype=torch.int32)
        state.age = torch.arange(n, dtype=torch.int32) * 2
        state.birth_step = torch.arange(n, dtype=torch.int32) * 3
        return state

    def test_reindex_permutation_carries_every_column(self) -> None:
        state = self._labelled(5)
        permutation = torch.tensor([3, 0, 4, 1, 2])
        state.on_reindex(permutation)
        self.assertEqual(len(state), 5)
        self.assertTrue(
            torch.allclose(state.error_ema, torch.tensor([0.3, 0.0, 0.4, 0.1, 0.2]))
        )
        self.assertTrue(
            torch.equal(state.anchor_index, torch.tensor([300, 0, 400, 100, 200]))
        )
        self.assertTrue(
            torch.equal(
                state.generation, torch.tensor([3, 0, 4, 1, 2], dtype=torch.int32)
            )
        )
        self.assertTrue(
            torch.equal(state.age, torch.tensor([6, 0, 8, 2, 4], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(
                state.birth_step, torch.tensor([9, 0, 12, 3, 6], dtype=torch.int32)
            )
        )
        # parent_id was the identity, so it must follow the permutation into
        # the NEW index space (row i pointing at itself).
        self.assertTrue(torch.equal(state.parent_id, torch.arange(5)))

    def test_prune_compacts_and_retargets_surviving_parents(self) -> None:
        state = self._labelled(6)
        state.parent_id = torch.tensor([0, 0, 1, 1, 4, 4])
        keep = torch.tensor([True, False, True, True, False, True])
        state.on_prune(keep)
        self.assertEqual(len(state), 4)
        for name in FIELD_NAMES:
            self.assertEqual(int(getattr(state, name).shape[0]), 4, name)
        self.assertTrue(
            torch.allclose(state.error_ema, torch.tensor([0.0, 0.2, 0.3, 0.5]))
        )
        self.assertTrue(
            torch.equal(state.anchor_index, torch.tensor([0, 200, 300, 500]))
        )
        # old parents [0, 1, 1, 4] -> new indices; 1 survived as 0's neighbour?
        # old index 0 -> new 0, old 2 -> new 1, old 3 -> new 2, old 5 -> new 3.
        # Parent 1 and 4 were pruned, so those children lose their parent.
        self.assertTrue(torch.equal(state.parent_id, torch.tensor([0, -1, -1, -1])))

    def test_prune_validates_the_mask(self) -> None:
        state = self._labelled(3)
        with self.assertRaises(ValueError):
            state.on_prune(torch.tensor([True, False]))
        with self.assertRaises(ValueError):
            state.on_prune(torch.tensor([1, 0, 1]))

    def test_prune_to_empty_keeps_every_column_consistent(self) -> None:
        state = self._labelled(3)
        state.on_prune(torch.zeros(3, dtype=torch.bool))
        self.assertEqual(len(state), 0)
        for name in FIELD_NAMES:
            self.assertEqual(int(getattr(state, name).shape[0]), 0, name)

    def test_resize_grows_and_shrinks_without_resetting(self) -> None:
        state = self._labelled(3)
        state.resize(5)
        self.assertTrue(
            torch.allclose(state.error_ema, torch.tensor([0.0, 0.1, 0.2, 1.0, 1.0]))
        )
        self.assertTrue(torch.equal(state.parent_id, torch.tensor([0, 1, 2, 3, 4])))
        self.assertTrue(torch.equal(state.anchor_index[3:], torch.tensor([-1, -1])))
        state.resize(2)
        self.assertTrue(torch.allclose(state.error_ema, torch.tensor([0.0, 0.1])))
        with self.assertRaises(ValueError):
            state.resize(-1)


class LifecycleCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        _requires_module(self)

    def _populated(self) -> "GaussianLifecycleState":
        state = GaussianLifecycleState(3)
        state.error_ema = torch.tensor([0.25, 0.5, 0.75])
        state.visibility_ema = torch.tensor([0.1, 0.2, 0.3])
        state.contribution_ema = torch.tensor([0.01, 0.02, 0.03])
        state.anchor_index = torch.tensor([7, -1, 12])
        state.anchor_confidence = torch.tensor([0.6, 0.0, 0.8])
        state.generation = torch.tensor([1, 0, 4], dtype=torch.int32)
        state.age = torch.tensor([100, 0, 55], dtype=torch.int32)
        state.parent_id = torch.tensor([2, 1, 0])
        state.birth_step = torch.tensor([300, 0, 900], dtype=torch.int32)
        state.current_step = 1234
        return state

    def test_state_dict_roundtrip_is_exact(self) -> None:
        source = self._populated()
        payload = source.state_dict()
        restored = GaussianLifecycleState(0)
        restored.load_state_dict(payload)
        self.assertEqual(len(restored), 3)
        self.assertEqual(restored.current_step, 1234)
        for name in FIELD_NAMES:
            original = getattr(source, name)
            copy = getattr(restored, name)
            self.assertEqual(copy.dtype, original.dtype, name)
            self.assertEqual(copy.device, original.device, name)
            self.assertTrue(torch.equal(copy, original), name)
            self.assertIsNot(copy, original, name)
        # The payload holds detached copies, not aliases of the live tensors.
        source.error_ema[0] = 42.0
        restored2 = GaussianLifecycleState(0)
        restored2.load_state_dict(payload)
        self.assertAlmostEqual(float(restored2.error_ema[0]), 0.25, places=6)

    def test_load_rejects_partial_stale_and_non_finite_payloads(self) -> None:
        good = self._populated().state_dict()
        missing = dict(good)
        missing.pop("anchor_index")
        stale_version = dict(good)
        stale_version["schema_version"] = 2
        ragged = dict(good)
        ragged["age"] = torch.zeros(2, dtype=torch.int32)
        non_finite = dict(good)
        non_finite["error_ema"] = torch.tensor([1.0, float("nan"), 1.0])
        two_dim = dict(good)
        two_dim["error_ema"] = torch.ones(3, 1)
        state = GaussianLifecycleState(3)
        for payload in (None, missing, stale_version, ragged, non_finite, two_dim):
            with self.subTest(payload=type(payload)):
                with self.assertRaises(ValueError):
                    state.load_state_dict(payload)
        with self.assertRaises(ValueError):
            state.load_state_dict(good, expected_count=4)
        state.load_state_dict(good, expected_count=3)


class ErrorScoreStateCompatibilityTests(unittest.TestCase):
    """The public ErrorScoreState API must be unchanged by the delegation."""

    def setUp(self) -> None:
        _requires_module(self)

    def test_scores_is_the_lifecycle_error_column(self) -> None:
        state = ErrorScoreState(3, ErrorScoreConfig(enabled=True))
        self.assertIs(state.scores, state.lifecycle.error_ema)
        state.scores[1] = 0.4  # in-place writes must reach the lifecycle
        self.assertAlmostEqual(float(state.lifecycle.error_ema[1]), 0.4, places=6)
        state.scores = torch.tensor([0.1, 0.2, 0.3])
        self.assertTrue(
            torch.allclose(state.lifecycle.error_ema, torch.tensor([0.1, 0.2, 0.3]))
        )
        self.assertEqual(len(state), 3)

    def test_assigning_a_different_length_realigns_every_column(self) -> None:
        state = ErrorScoreState(2, ErrorScoreConfig(enabled=True))
        state.scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(len(state), 4)
        self.assertEqual(len(state.lifecycle), 4)
        for name in FIELD_NAMES:
            self.assertEqual(int(getattr(state.lifecycle, name).shape[0]), 4, name)

    def test_sampling_weights_match_the_pre_migration_formula(self) -> None:
        config = ErrorScoreConfig(
            enabled=True, score_power=0.4, min_score_floor=1e-3
        )
        state = ErrorScoreState(5, config)
        scores = torch.tensor([0.0, 1e-9, 0.25, 0.8, 3.0])
        state.scores = scores.clone()
        opacities = torch.tensor([0.2, 0.4, 0.6, 0.8, 0.05])
        # Literal re-implementation of the pre-WP-1 body of sampling_weights.
        legacy = opacities * scores.clamp_min(1e-3) ** 0.4
        self.assertTrue(torch.equal(state.sampling_weights(opacities), legacy))

    def test_checkpoint_roundtrip_carries_the_lifecycle(self) -> None:
        source = ErrorScoreState(3, ErrorScoreConfig(enabled=True))
        source.scores = torch.tensor([0.15, 0.5, 0.95])
        source.lifecycle.anchor_index = torch.tensor([4, -1, 9])
        source.lifecycle.generation = torch.tensor([2, 0, 1], dtype=torch.int32)
        source.lifecycle.parent_id = torch.tensor([1, 1, 2])
        payload = source.checkpoint_state()
        self.assertEqual(payload["schema_version"], 1)

        restored = ErrorScoreState(3, ErrorScoreConfig(enabled=True))
        restored.restore_checkpoint_state(payload, expected_count=3)
        self.assertTrue(torch.equal(restored.scores, source.scores))
        self.assertTrue(
            torch.equal(restored.lifecycle.anchor_index, torch.tensor([4, -1, 9]))
        )
        self.assertTrue(
            torch.equal(
                restored.lifecycle.generation, torch.tensor([2, 0, 1], dtype=torch.int32)
            )
        )
        self.assertTrue(
            torch.equal(restored.lifecycle.parent_id, torch.tensor([1, 1, 2]))
        )

    def test_legacy_checkpoint_without_lifecycle_still_resumes(self) -> None:
        # Payload written before WP-1: scores only, no "lifecycle" key.
        legacy_payload = {
            "schema_version": 1,
            "scores": torch.tensor([0.3, 0.6, 0.9]),
        }
        restored = ErrorScoreState(3, ErrorScoreConfig(enabled=True))
        restored.restore_checkpoint_state(legacy_payload, expected_count=3)
        self.assertTrue(
            torch.allclose(restored.scores, torch.tensor([0.3, 0.6, 0.9]))
        )
        self.assertEqual(len(restored.lifecycle), 3)
        self.assertTrue(
            torch.equal(restored.lifecycle.anchor_index, torch.full((3,), -1))
        )
        self.assertTrue(torch.equal(restored.lifecycle.parent_id, torch.arange(3)))

    def test_reset_drops_history_for_a_brand_new_cloud(self) -> None:
        state = ErrorScoreState(3, ErrorScoreConfig(enabled=True))
        state.scores = torch.tensor([0.1, 0.2, 0.3])
        state.lifecycle.anchor_index = torch.tensor([5, 6, 7])
        state.reset(4)
        self.assertEqual(len(state), 4)
        self.assertTrue(torch.equal(state.scores, torch.ones(4)))
        self.assertTrue(torch.equal(state.lifecycle.anchor_index, torch.full((4,), -1)))
        self.assertTrue(torch.equal(state.lifecycle.parent_id, torch.arange(4)))

    def test_on_step_advances_the_lifecycle_clock(self) -> None:
        state = ErrorScoreState(2, ErrorScoreConfig(enabled=True))
        state.on_step(11)
        state.on_step(12)
        self.assertTrue(
            torch.equal(state.lifecycle.age, torch.full((2,), 2, dtype=torch.int32))
        )
        self.assertEqual(state.lifecycle.current_step, 12)


class StrategyLifecycleWiringTests(unittest.TestCase):
    """End-to-end index consistency across a real (CPU) sample_add / relocate."""

    def setUp(self) -> None:
        _requires_module(self)

    def _params_and_optimizers(self, n: int):
        torch.manual_seed(11)
        params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.rand(n, 3)),
                "scales": torch.nn.Parameter(torch.rand(n, 3) - 2.0),
                "quats": torch.nn.Parameter(torch.rand(n, 4)),
                "opacities": torch.nn.Parameter(torch.full((n,), 0.5).logit()),
            }
        )
        return params, _make_optimizers(params, warm=True)

    def test_sample_add_keeps_params_optimizer_and_lifecycle_aligned(self) -> None:
        n = 40
        params, optimizers = self._params_and_optimizers(n)
        state = ErrorScoreState(n, ErrorScoreConfig(enabled=True))
        state.scores = torch.linspace(0.05, 0.95, n)
        state.lifecycle.anchor_index = torch.arange(n) * 10
        state.lifecycle.anchor_confidence = torch.linspace(0.0, 1.0, n)
        state.lifecycle.generation = torch.full((n,), 2, dtype=torch.int32)
        state.lifecycle.age = torch.full((n,), 500, dtype=torch.int32)
        scores_before = state.scores.clone()
        means_before = params["means"].detach().clone()
        exp_avg_before = optimizers["means"].state[params["means"]]["exp_avg"].clone()

        strategy = ErrorWeightedMCMCStrategy(
            cap_max=10_000, score_state=state, error_config=state.config
        )
        strategy.current_step = 4100
        binoms = strategy.initialize_state()["binoms"]
        parents = torch.tensor([3, 7])
        with mock.patch(
            f"{_MODULE}.compute_relocation", side_effect=_fake_compute_relocation
        ), mock.patch(f"{_MODULE}._multinomial_sample", return_value=parents):
            n_gs = strategy._add_new_gs(params, optimizers, binoms)

        self.assertEqual(n_gs, 2)
        self.assertEqual(len(params["means"]), n + 2)
        self.assertEqual(len(state), n + 2)
        for name in FIELD_NAMES:
            self.assertEqual(
                int(getattr(state.lifecycle, name).shape[0]), n + 2, name
            )
        # Parameter index space: the two appended rows are copies of parents.
        self.assertTrue(
            torch.equal(params["means"].detach()[n:], means_before[parents])
        )
        self.assertTrue(
            torch.equal(params["means"].detach()[:n], means_before)
        )
        # Optimizer state: survivors preserved, new slots zeroed by gsplat.
        exp_avg_after = optimizers["means"].state[params["means"]]["exp_avg"]
        self.assertEqual(tuple(exp_avg_after.shape), (n + 2, 3))
        self.assertTrue(torch.equal(exp_avg_after[:n], exp_avg_before))
        self.assertTrue(torch.equal(exp_avg_after[n:], torch.zeros(2, 3)))
        # Error state: survivors bit-identical, children inherit the parents.
        self.assertTrue(torch.equal(state.scores[:n], scores_before))
        self.assertTrue(torch.equal(state.scores[n:], scores_before[parents]))
        # Anchor and parent-child identity.
        self.assertTrue(
            torch.equal(state.lifecycle.anchor_index[n:], parents * 10)
        )
        self.assertTrue(
            torch.equal(state.lifecycle.parent_id[n:], parents)
        )
        self.assertTrue(
            torch.equal(
                state.lifecycle.generation[n:], torch.full((2,), 3, dtype=torch.int32)
            )
        )
        self.assertTrue(
            torch.equal(
                state.lifecycle.birth_step[n:],
                torch.full((2,), 4100, dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(state.lifecycle.age[n:], torch.zeros(2, dtype=torch.int32))
        )

    def test_relocate_moves_state_from_source_into_the_dead_slot(self) -> None:
        opacities = torch.tensor([0.5, 0.001, 0.8, 0.002, 0.6])
        params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.arange(15, dtype=torch.float32).reshape(5, 3)),
                "scales": torch.nn.Parameter(torch.full((5, 3), -2.0)),
                "quats": torch.nn.Parameter(torch.zeros(5, 4)),
                "opacities": torch.nn.Parameter(opacities.logit()),
            }
        )
        state = ErrorScoreState(5, ErrorScoreConfig(enabled=True))
        state.scores = torch.tensor([0.2, 0.4, 0.9, 0.1, 0.7])
        state.lifecycle.anchor_index = torch.tensor([0, 10, 20, 30, 40])
        optimizers = _make_optimizers(params, warm=True)
        strategy = ErrorWeightedMCMCStrategy(
            score_state=state, error_config=state.config
        )
        strategy.current_step = 2500
        binoms = strategy.initialize_state()["binoms"]
        # dead = [1, 3]; alive = [0, 2, 4]; pick alive positions [1, 2] -> [2, 4]
        with mock.patch(
            f"{_MODULE}.compute_relocation", side_effect=_fake_compute_relocation
        ), mock.patch(
            f"{_MODULE}._multinomial_sample", return_value=torch.tensor([1, 2])
        ):
            n_gs = strategy._relocate_gs(params, optimizers, binoms)

        self.assertEqual(n_gs, 2)
        self.assertEqual(len(state), 5)
        # Parameters: dead slots hold copies of their sources.
        means = params["means"].detach()
        self.assertTrue(torch.equal(means[1], means[2]))
        self.assertTrue(torch.equal(means[3], means[4]))
        # Lifecycle followed the same pairing.
        self.assertTrue(
            torch.allclose(state.scores, torch.tensor([0.2, 0.9, 0.9, 0.7, 0.7]))
        )
        self.assertTrue(
            torch.equal(state.lifecycle.anchor_index, torch.tensor([0, 20, 20, 40, 40]))
        )
        self.assertTrue(
            torch.equal(state.lifecycle.parent_id, torch.tensor([0, 2, 2, 4, 4]))
        )
        self.assertTrue(
            torch.equal(
                state.lifecycle.birth_step,
                torch.tensor([0, 2500, 0, 2500, 0], dtype=torch.int32),
            )
        )

    def test_step_post_backward_records_the_step_for_births(self) -> None:
        state = ErrorScoreState(4, ErrorScoreConfig(enabled=True))
        strategy = ErrorWeightedMCMCStrategy(
            score_state=state, error_config=state.config
        )
        with mock.patch("gsplat.strategy.mcmc.MCMCStrategy.step_post_backward"):
            strategy.step_post_backward(
                params={}, optimizers={}, state={}, step=777, info={}, lr=1e-3
            )
        self.assertEqual(strategy.current_step, 777)


class DensifyPreservesHistoryTests(unittest.TestCase):
    """Multi-step EMA -> densify -> more EMA, the long-training failure mode."""

    def setUp(self) -> None:
        _requires_module(self)

    def test_accumulated_ema_survives_densification(self) -> None:
        n = 40
        decay = 0.9
        config = ErrorScoreConfig(enabled=True, ema_decay=decay)
        state = ErrorScoreState(n, config)
        means2d = torch.zeros(n, 2)
        radii = torch.ones(n, dtype=torch.int64)
        observed_error = 0.2
        pixel_error = torch.full((4, 4), observed_error)
        steps = 60
        for _ in range(steps):
            state.update(means2d, radii, pixel_error, height=4, width=4)
        converged = decay**steps + (1.0 - decay**steps) * observed_error
        self.assertAlmostEqual(float(state.scores[0]), converged, places=5)
        # Sanity: the EMA has actually moved far away from the 1.0 default.
        self.assertLess(float(state.scores.max()), 0.25)
        scores_before = state.scores.clone()

        params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.zeros(n, 3)),
                "scales": torch.nn.Parameter(torch.full((n, 3), -2.0)),
                "quats": torch.nn.Parameter(torch.zeros(n, 4)),
                "opacities": torch.nn.Parameter(torch.full((n,), 0.5).logit()),
            }
        )
        optimizers = _make_optimizers(params, warm=True)
        strategy = ErrorWeightedMCMCStrategy(
            cap_max=10_000, score_state=state, error_config=config
        )
        strategy.current_step = 3000
        binoms = strategy.initialize_state()["binoms"]
        parents = torch.tensor([0, 1])
        with mock.patch(
            f"{_MODULE}.compute_relocation", side_effect=_fake_compute_relocation
        ), mock.patch(f"{_MODULE}._multinomial_sample", return_value=parents):
            strategy._add_new_gs(params, optimizers, binoms)

        # Under the old resize()-resets-to-ones behaviour every entry here
        # would be exactly 1.0 and both assertions below would fail.
        self.assertTrue(torch.equal(state.scores[:n], scores_before))
        self.assertLess(float(state.scores.max()), 0.25)
        self.assertTrue(torch.equal(state.scores[n:], scores_before[parents]))

        # Accumulation continues from the preserved value, not from 1.0.
        grown = len(state)
        state.update(
            torch.zeros(grown, 2),
            torch.ones(grown, dtype=torch.int64),
            pixel_error,
            height=4,
            width=4,
        )
        expected = decay * float(scores_before[0]) + (1.0 - decay) * observed_error
        self.assertAlmostEqual(float(state.scores[0]), expected, places=6)


if __name__ == "__main__":
    unittest.main()
