"""Pure-torch pieces of ``densification_gradient_source="rgb_only"``.

The trainer optimizes the TOTAL loss but wants the classic densification
criterion to score Gaussians by the gradient of the photometric loss alone, as
Kerbl et al. trained. Both facts have to hold in one step, and the rasterizer
gives them different persistence:

* ``means2d.grad`` ACCUMULATES across backward passes (it is a retained
  non-leaf gradient), so a second backward adds the geometry terms to it;
* ``means2d.absgrad`` is OVERWRITTEN by gsplat on every rasterizer backward
  (``means2d.absgrad = v_means2d_abs``), so a second backward replaces it.

The only representation that survives both is a snapshot taken between the
passes. These helpers hold that snapshot logic and the two-pass driver so they
can be exercised on CPU tensors without the CUDA rasterizer; the backend and
trainer call them rather than re-implementing the sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class CriterionGradientSnapshot:
    """Densification-criterion gradients captured after the photometric pass."""

    grad: Any | None
    absgrad: Any | None


def snapshot_criterion_gradients(
    means2d: Any, *, require_absgrad: bool
) -> CriterionGradientSnapshot:
    """Clone ``means2d.grad``/``.absgrad`` right after the photometric backward.

    Fails closed: a strategy that scores from a gradient this pass did not
    produce would otherwise densify nothing, silently, for the whole run.
    """
    grad = means2d.grad
    snapshot_grad = None if grad is None else grad.detach().clone()
    absgrad = getattr(means2d, "absgrad", None)
    snapshot_absgrad = None if absgrad is None else absgrad.detach().clone()
    if require_absgrad and snapshot_absgrad is None:
        raise RuntimeError(
            "absgrad strategy is active but the photometric backward "
            "produced no means2d.absgrad"
        )
    if snapshot_grad is None and snapshot_absgrad is None:
        raise RuntimeError(
            "photometric backward left no gradient on means2d; "
            "was strategy_pre_step skipped?"
        )
    return CriterionGradientSnapshot(grad=snapshot_grad, absgrad=snapshot_absgrad)


def restore_criterion_gradients(
    means2d: Any, snapshot: CriterionGradientSnapshot
) -> None:
    """Put the photometric-only gradients back for the strategy to read.

    ``grad`` is always assigned (it may legitimately be None under an
    absgrad-only strategy); ``absgrad`` is only assigned when the snapshot
    holds one, so a plain-gradient strategy never grows a stale attribute.
    """
    means2d.grad = snapshot.grad
    if snapshot.absgrad is not None:
        means2d.absgrad = snapshot.absgrad


def photometric_growth_loss(
    *,
    l1: Any,
    ssim: Any,
    rgb_l1_weight: float,
    rgb_ssim_weight: float,
    rgb_gradient_weight: float,
    rgb_gradient_l1: Any | None,
    lidar_rgb_l1_weight: float,
    lidar_rgb_l1: Any | None,
) -> Any:
    """The RGB-only loss the densification criterion differentiates.

    Same terms, same weights and same summation order as the photometric part
    of the training loss, so ``total - this`` is exactly the non-photometric
    remainder up to float rounding. The gradient audit reuses it so its "rgb"
    component is the very tensor the growth signal is taken from.
    """
    loss = rgb_l1_weight * l1 + rgb_ssim_weight * ssim
    if rgb_gradient_l1 is not None:
        loss = loss + rgb_gradient_weight * rgb_gradient_l1
    if lidar_rgb_l1 is not None:
        loss = loss + lidar_rgb_l1_weight * lidar_rgb_l1
    return loss


def split_backward_for_growth_signal(
    *,
    total_loss: Any,
    growth_loss: Any,
    isolate: Callable[[], None],
    restore: Callable[[], None],
) -> None:
    """Two backward passes: growth signal from ``growth_loss``, optimizer from total.

    1. ``growth_loss.backward(retain_graph=True)`` - leaves receive the
       photometric gradient, means2d receives the criterion gradient;
    2. ``isolate()`` snapshots the means2d criterion gradients;
    3. ``(total_loss - growth_loss).backward()`` - leaves accumulate the
       remainder, so after both passes they hold the total-loss gradient;
    4. ``restore()`` puts the photometric-only means2d gradients back.

    The remainder is formed by subtraction rather than by re-summing the
    other terms so the caller cannot drift from the loss it actually
    optimizes: whatever was in ``total_loss`` reaches the leaves.
    """
    remainder = total_loss - growth_loss
    growth_loss.backward(retain_graph=True)
    isolate()
    if remainder.requires_grad:
        remainder.backward()
    restore()
