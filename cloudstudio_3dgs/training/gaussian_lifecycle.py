# SPDX-License-Identifier: Apache-2.0
"""Per-Gaussian lifecycle state carried across MCMC relocation and densification.

Everything in this module is *bookkeeping*: it stores one row per Gaussian,
kept at exactly the same length and in exactly the same order as the parameter
tensors (``means``/``scales``/``quats``/``opacities``/colors) and the optimizer
state that gsplat re-orders alongside them. No scoring formula, loss term or
sampling decision lives here.

Why it exists
-------------
Before this module, the only per-Gaussian side state was
:class:`~cloudstudio_3dgs.training.error_weighted_mcmc.ErrorScoreState.scores`,
and its ``resize()`` **reset every score to 1.0** whenever densification
changed the Gaussian count. With MCMC growing the cloud every
``refine_every`` steps, a multi-thousand-step multi-view error EMA was thrown
away on a fixed cadence and the error-weighted sampler repeatedly degraded to
plain opacity MCMC. The lifecycle operations below are all *incremental*:
grow appends, relocate overwrites only the dead slots, prune/reindex gather.
No operation ever resets a surviving entry.

Index contract
--------------
For a cloud of ``N`` Gaussians every field is a 1-D tensor of length ``N`` on
the same device, so ``field[i]`` always describes ``params["means"][i]``.
The lifecycle operations mirror the four ways gsplat mutates that index space:

* ``on_grow``     — ``sample_add``: ``k`` rows appended, cloned from parents.
* ``on_relocate`` — ``relocate``: dead rows overwritten by their source rows.
* ``on_prune``    — boolean keep-mask compaction.
* ``on_reindex``  — arbitrary gather/permutation.

Fields
------
``error_ema``          image-space error EMA (the migrated ``scores`` tensor).
``visibility_ema``     EMA of per-view visibility; reserved, maintained only.
``contribution_ema``   EMA of per-pixel alpha contribution; reserved for WP-2,
                       maintained but not consumed anywhere yet.
``anchor_index``       index into the LiDAR normal field, ``-1`` when unanchored;
                       reserved so a future anchor refresh can follow children
                       and relocations instead of invalidating the whole cache.
``anchor_confidence``  planarity/confidence of that anchor.
``generation``         0 for the initialization cloud, ``parent + 1`` for every
                       Gaussian born from a split/clone/relocation.
``age``                steps since birth.
``parent_id``          index of the Gaussian this one was cloned from *at birth
                       time*; own index for roots. Remapped by ``on_prune`` /
                       ``on_reindex`` while the parent survives, ``-1`` after
                       the parent is pruned away.
``birth_step``         training step at which the row was (re)born.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
from torch import Tensor

# name, dtype, scalar default for freshly created rows
_FIELD_SPECS: tuple[tuple[str, torch.dtype, float], ...] = (
    ("error_ema", torch.float32, 1.0),
    ("visibility_ema", torch.float32, 0.0),
    ("contribution_ema", torch.float32, 0.0),
    ("anchor_index", torch.int64, -1.0),
    ("anchor_confidence", torch.float32, 0.0),
    ("generation", torch.int32, 0.0),
    ("age", torch.int32, 0.0),
    ("parent_id", torch.int64, -1.0),
    ("birth_step", torch.int32, 0.0),
)

FIELD_NAMES: tuple[str, ...] = tuple(name for name, _dtype, _default in _FIELD_SPECS)

_FIELD_DTYPES: dict[str, torch.dtype] = {
    name: dtype for name, dtype, _default in _FIELD_SPECS
}

_SCHEMA_VERSION = 1


class GaussianLifecycleState:
    """Row-per-Gaussian lifecycle bookkeeping aligned with the parameter tensors."""

    def __init__(
        self,
        num_gaussians: int,
        device: torch.device | str | None = None,
    ) -> None:
        count = int(num_gaussians)
        if count < 0:
            raise ValueError("num_gaussians must be non-negative")
        resolved = torch.device(device) if device is not None else torch.device("cpu")
        for name, value in self._fresh_rows(count, resolved, root=True).items():
            setattr(self, name, value)
        self.current_step: int = 0

    # -- construction helpers ------------------------------------------------

    @staticmethod
    def _fresh_rows(
        count: int,
        device: torch.device,
        *,
        root: bool,
        offset: int = 0,
    ) -> dict[str, Tensor]:
        """Default rows for ``count`` new Gaussians with no known parent.

        ``root=True`` makes each row its own parent (the initialization cloud
        and any length-alignment growth), which keeps ``parent_id`` a valid
        index instead of a sentinel for Gaussians that were never cloned.
        """
        rows: dict[str, Tensor] = {}
        for name, dtype, default in _FIELD_SPECS:
            rows[name] = torch.full((count,), default, dtype=dtype, device=device)
        if root:
            rows["parent_id"] = torch.arange(
                offset, offset + count, dtype=torch.int64, device=device
            )
        return rows

    # -- basic accessors -----------------------------------------------------

    def __len__(self) -> int:
        return int(self.error_ema.shape[0])

    @property
    def device(self) -> torch.device:
        return self.error_ema.device

    def fields(self) -> dict[str, Tensor]:
        """Return the live field tensors keyed by name (not copies)."""
        return {name: getattr(self, name) for name in FIELD_NAMES}

    def to(self, device: torch.device | str) -> "GaussianLifecycleState":
        """Move every field onto ``device`` in place and return self."""
        target = torch.device(device)
        for name in FIELD_NAMES:
            setattr(self, name, getattr(self, name).to(target))
        return self

    # -- index validation ----------------------------------------------------

    def _as_index(self, indices: Any, *, name: str, upper: int) -> Tensor:
        idx = torch.as_tensor(indices, device=self.device)
        if idx.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        if idx.dtype != torch.int64:
            if idx.is_floating_point() or idx.dtype == torch.bool:
                raise ValueError(f"{name} must be an integer index tensor")
            idx = idx.long()
        idx = idx.reshape(-1)
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= upper):
            raise ValueError(f"{name} contains out-of-range indices")
        return idx

    # -- lifecycle operations ------------------------------------------------

    @torch.no_grad()
    def on_grow(
        self,
        parent_indices: Any,
        step: int,
        birth_factor: float = 1.0,
    ) -> Tensor:
        """Append one row per entry of ``parent_indices``, inheriting from it.

        The children are appended in the same order gsplat's ``sample_add``
        concatenates the cloned parameters, so row ``old_n + j`` describes the
        clone of ``parent_indices[j]``. Inherited: ``error_ema`` (scaled by
        ``birth_factor``), ``anchor_index``, ``anchor_confidence``. Derived:
        ``generation = parent + 1``, ``parent_id = parent`` (the pre-growth
        global index), ``birth_step = step``, ``age = 0``. The visibility and
        contribution EMAs start at zero because a brand new Gaussian has not
        been observed yet.

        Returns the indices of the newly appended rows.
        """
        factor = float(birth_factor)
        if factor != factor or factor in (float("inf"), float("-inf")) or factor < 0.0:
            raise ValueError("birth_factor must be finite and non-negative")
        old_n = len(self)
        parents = self._as_index(parent_indices, name="parent_indices", upper=old_n)
        added = int(parents.numel())
        if added == 0:
            self.current_step = int(step)
            return torch.empty(0, dtype=torch.int64, device=self.device)

        step_value = int(step)
        children = self._fresh_rows(added, self.device, root=False)
        children["error_ema"] = self.error_ema[parents] * factor
        children["anchor_index"] = self.anchor_index[parents].clone()
        children["anchor_confidence"] = self.anchor_confidence[parents].clone()
        children["generation"] = self.generation[parents] + 1
        children["parent_id"] = parents.clone()
        children["birth_step"] = torch.full(
            (added,), step_value, dtype=torch.int32, device=self.device
        )
        for name in FIELD_NAMES:
            merged = torch.cat(
                [getattr(self, name), children[name].to(_FIELD_DTYPES[name])]
            )
            setattr(self, name, merged)
        self.current_step = step_value
        return torch.arange(old_n, old_n + added, dtype=torch.int64, device=self.device)

    @torch.no_grad()
    def on_relocate(self, dead_indices: Any, source_indices: Any, step: int) -> None:
        """Overwrite dead rows with the state of the Gaussians they now clone.

        MCMC relocation copies the source Gaussian's geometry, opacity and
        color into the dead slot, so the slot's lifecycle state must follow the
        *source*, not the corpse it replaced: keeping the dead row's error EMA
        would attribute a stale, unrelated error history to freshly teleported
        geometry. The source rows themselves are left untouched.
        """
        count = len(self)
        dead = self._as_index(dead_indices, name="dead_indices", upper=count)
        source = self._as_index(source_indices, name="source_indices", upper=count)
        if int(dead.numel()) != int(source.numel()):
            raise ValueError("dead_indices and source_indices must have equal length")
        if dead.numel() == 0:
            self.current_step = int(step)
            return

        # Gather first: a source row must never be read after it was written.
        inherited_error = self.error_ema[source].clone()
        inherited_anchor = self.anchor_index[source].clone()
        inherited_confidence = self.anchor_confidence[source].clone()
        inherited_generation = (self.generation[source] + 1).to(torch.int32)

        self.error_ema[dead] = inherited_error
        self.anchor_index[dead] = inherited_anchor
        self.anchor_confidence[dead] = inherited_confidence
        self.generation[dead] = inherited_generation
        self.parent_id[dead] = source
        self.birth_step[dead] = int(step)
        self.age[dead] = 0
        self.visibility_ema[dead] = 0.0
        self.contribution_ema[dead] = 0.0
        self.current_step = int(step)

    @torch.no_grad()
    def on_prune(self, keep_mask: Any) -> None:
        """Compact every field with a boolean keep-mask over the current rows."""
        mask = torch.as_tensor(keep_mask, device=self.device).reshape(-1)
        if mask.dtype != torch.bool:
            raise ValueError("keep_mask must be a boolean tensor")
        if int(mask.shape[0]) != len(self):
            raise ValueError("keep_mask length does not match the lifecycle state")
        self.on_reindex(mask.nonzero(as_tuple=True)[0])

    @torch.no_grad()
    def on_reindex(self, index_map: Any) -> None:
        """Gather every field so that new row ``i`` holds old row ``index_map[i]``.

        ``parent_id`` is translated into the new index space: a parent that
        survived keeps pointing at it, a parent that disappeared becomes ``-1``.
        Every other field is a pure gather.
        """
        old_n = len(self)
        gather = self._as_index(index_map, name="index_map", upper=old_n)
        new_n = int(gather.numel())
        inverse = torch.full((old_n,), -1, dtype=torch.int64, device=self.device)
        if new_n:
            inverse[gather] = torch.arange(
                new_n, dtype=torch.int64, device=self.device
            )
        old_parent = self.parent_id
        for name in FIELD_NAMES:
            setattr(self, name, getattr(self, name)[gather].clone())
        if new_n:
            picked = old_parent[gather]
            valid = picked >= 0
            translated = torch.where(
                valid, inverse[picked.clamp_min(0)], torch.full_like(picked, -1)
            )
            self.parent_id = translated
        else:
            self.parent_id = torch.empty(0, dtype=torch.int64, device=self.device)

    @torch.no_grad()
    def on_step(self, step: int, indices: Any | None = None) -> None:
        """Advance ages by one step (all rows, or just ``indices``)."""
        if indices is None:
            if len(self):
                self.age += 1
        else:
            idx = self._as_index(indices, name="indices", upper=len(self))
            if idx.numel():
                self.age[idx] += 1
        self.current_step = int(step)

    @torch.no_grad()
    def resize(self, new_count: int) -> None:
        """Align the length to ``new_count`` while preserving existing rows.

        Growth appends default rows (root ``parent_id``, ``error_ema`` 1.0);
        shrinking truncates the tail. This is the fallback used when the exact
        parent/child mapping of a refinement is not observable (e.g. gsplat's
        upstream ``sample_add`` on the non-weighted code path); prefer
        :meth:`on_grow` / :meth:`on_prune` whenever the mapping is known.
        """
        target = int(new_count)
        if target < 0:
            raise ValueError("new_count must be non-negative")
        current = len(self)
        if target == current:
            return
        if target < current:
            self.on_reindex(
                torch.arange(target, dtype=torch.int64, device=self.device)
            )
            return
        added = target - current
        rows = self._fresh_rows(added, self.device, root=True, offset=current)
        for name in FIELD_NAMES:
            setattr(self, name, torch.cat([getattr(self, name), rows[name]]))

    # -- checkpointing -------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Serialize every field for checkpointing (dtype and device preserved)."""
        payload: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "count": len(self),
            "current_step": int(self.current_step),
        }
        for name in FIELD_NAMES:
            payload[name] = getattr(self, name).detach().clone()
        return payload

    @torch.no_grad()
    def load_state_dict(
        self,
        payload: Any,
        *,
        expected_count: int | None = None,
    ) -> None:
        """Restore fields from :meth:`state_dict`, rejecting stale/partial payloads."""
        if not isinstance(payload, dict):
            raise ValueError("lifecycle state must be a mapping")
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("lifecycle state has an unsupported schema version")
        restored: dict[str, Tensor] = {}
        count: Optional[int] = None
        for name in FIELD_NAMES:
            tensor = payload.get(name)
            if not isinstance(tensor, Tensor) or tensor.dim() != 1:
                raise ValueError(f"lifecycle field {name!r} must be a 1-D tensor")
            if count is None:
                count = int(tensor.shape[0])
            elif int(tensor.shape[0]) != count:
                raise ValueError("lifecycle fields have inconsistent lengths")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"lifecycle field {name!r} contains non-finite values")
            restored[name] = tensor
        assert count is not None
        stored_count = payload.get("count")
        if stored_count is not None and int(stored_count) != count:
            raise ValueError("lifecycle count does not match the stored fields")
        if expected_count is not None and count != int(expected_count):
            raise ValueError(
                "checkpoint lifecycle count does not match restored Gaussians"
            )
        device = self.device
        for name, tensor in restored.items():
            setattr(
                self,
                name,
                tensor.detach().to(device=device, dtype=_FIELD_DTYPES[name]).clone(),
            )
        self.current_step = int(payload.get("current_step", 0))


def lifecycle_fields() -> Iterable[str]:
    """Field names in their canonical order (stable across checkpoints)."""
    return FIELD_NAMES
