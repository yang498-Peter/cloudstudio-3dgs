"""Direct gsplat API adapter for raw-fisheye 3DGUT and MCMC."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any

from cloudstudio_3dgs.training.runtime_evidence import (
    audit_loaded_mcmc_runtime,
    build_mcmc_step_event,
    require_full_mcmc_runtime,
    snapshot_gaussians,
)


def verify_gsplat_runtime(lock_path: Path) -> dict[str, Any]:
    """Require the locked version and, for VCS installs, the exact clean commit."""
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if lock.get("patch") not in (None, ""):
        raise ValueError("CloudStudio trainer lock must not require a gsplat source patch")
    if lock.get("source_policy") != "clean_vcs_commit":
        raise ValueError("CloudStudio trainer currently requires clean_vcs_commit provenance")
    installed_version = importlib.metadata.version("gsplat")
    if installed_version != str(lock["version"]):
        raise RuntimeError(
            f"gsplat version mismatch: expected {lock['version']}, got {installed_version}"
        )
    import gsplat

    module_path = Path(gsplat.__file__).resolve()
    repository = next((parent for parent in module_path.parents if (parent / ".git").exists()), None)
    evidence: dict[str, Any] = {
        "package": "gsplat",
        "version": installed_version,
        "module_path": str(module_path),
        "locked_commit": str(lock["commit"]),
    }
    if repository is None:
        raise RuntimeError("gsplat must be imported from a clean checkout at the locked commit")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != str(lock["commit"]):
        raise RuntimeError(f"gsplat commit mismatch: expected {lock['commit']}, got {head}")
    if dirty:
        raise RuntimeError("gsplat checkout has local modifications")
    evidence.update(
        {
            "source_kind": "clean_vcs",
            "repository_path": str(repository),
            "commit": head,
            "clean": True,
        }
    )
    return evidence


class GsplatBackend:
    """Small adapter whose only renderer dependency is the public gsplat API."""

    # Default for instances constructed without __init__ in contract tests.
    error_score_state: Any = None

    def __init__(
        self,
        *,
        device: str,
        cap_max: int,
        lock_path: Path,
        mcmc_config: dict[str, Any] | None = None,
        error_score_config: Any | None = None,
    ) -> None:
        self.runtime = verify_gsplat_runtime(lock_path)
        import torch
        from gsplat import rasterization

        from cloudstudio_3dgs.training.error_weighted_mcmc import (
            ErrorScoreConfig,
            ErrorScoreState,
            ErrorWeightedMCMCStrategy,
        )

        self.torch = torch
        self.rasterization = rasterization
        self.device = device
        if error_score_config is None:
            error_score_config = ErrorScoreConfig()
        # Gaussian count is unknown until initialize(); the state starts empty
        # and the strategy falls back to pure-opacity sampling until resized.
        self.error_score_state = (
            ErrorScoreState(0, error_score_config, device=device)
            if error_score_config.enabled
            else None
        )
        self.strategy = ErrorWeightedMCMCStrategy(
            cap_max=cap_max,
            verbose=False,
            score_state=self.error_score_state,
            error_config=error_score_config,
            **({} if mcmc_config is None else mcmc_config),
        )
        operator_report = audit_loaded_mcmc_runtime(self.runtime)
        self.runtime["mcmc_operator_report"] = operator_report
        noise_stop = -1 if mcmc_config is None else int(
            mcmc_config.get("noise_injection_stop_iter", -1)
        )
        if noise_stop != 0:
            require_full_mcmc_runtime(operator_report)

    SH_C0 = 0.28209479177387814

    def initialize(
        self,
        xyz: Any,
        rgb: Any,
        *,
        init_scale_m: float | None = None,
        init_scales_m: Any | None = None,
        learning_rates: dict[str, float],
        color_model: str = "rgb_sigmoid",
        sh_degree: int = 2,
    ) -> tuple[Any, dict[str, Any], Any]:
        torch = self.torch
        if color_model not in ("rgb_sigmoid", "sh"):
            raise ValueError("color_model must be 'rgb_sigmoid' or 'sh'")
        if color_model == "sh" and not 0 <= int(sh_degree) <= 3:
            raise ValueError("sh_degree must be within [0, 3]")
        self.color_model = color_model
        self.sh_degree = int(sh_degree)
        points = torch.as_tensor(xyz, dtype=torch.float32, device=self.device)
        colors = torch.as_tensor(rgb, dtype=torch.float32, device=self.device) / 255.0
        if len(points) < 4:
            raise ValueError("at least four initialization points are required")
        if (init_scale_m is None) == (init_scales_m is None):
            raise ValueError("provide exactly one of init_scale_m or init_scales_m")
        if init_scales_m is None:
            if init_scale_m is None or init_scale_m <= 0.0:
                raise ValueError("init_scale_m must be positive")
            scales_m = torch.full(
                (len(points), 3), float(init_scale_m), device=self.device
            )
        else:
            scales_m = torch.as_tensor(
                init_scales_m, dtype=torch.float32, device=self.device
            )
            if scales_m.ndim == 1 and scales_m.shape == (len(points),):
                scales_m = scales_m[:, None].repeat(1, 3)
            if scales_m.shape != (len(points), 3):
                raise ValueError("init_scales_m must have shape [N] or [N, 3]")
            if not bool(torch.isfinite(scales_m).all()) or not bool((scales_m > 0.0).all()):
                raise ValueError("init_scales_m must be finite and positive")
        quaternions = torch.zeros((len(points), 4), device=self.device)
        quaternions[:, 0] = 1.0
        entries = {
            "means": torch.nn.Parameter(points),
            "scales": torch.nn.Parameter(scales_m.log()),
            "quats": torch.nn.Parameter(quaternions),
            "opacities": torch.nn.Parameter(
                torch.full((len(points),), 0.1, device=self.device).logit()
            ),
        }
        if color_model == "sh":
            # Standard spherical-harmonics color: the DC coefficient carries the
            # point color ((c - 0.5) / C0) and the view-dependent bands start at
            # zero. Rasterization converts SH to RGB when sh_degree is passed.
            coefficient_count = (self.sh_degree + 1) ** 2
            sh0 = ((colors - 0.5) / self.SH_C0)[:, None, :]
            entries["sh0"] = torch.nn.Parameter(sh0)
            entries["shN"] = torch.nn.Parameter(
                torch.zeros(
                    (len(points), coefficient_count - 1, 3), device=self.device
                )
            )
        else:
            entries["colors"] = torch.nn.Parameter(
                colors.clamp(1e-4, 1.0 - 1e-4).logit()
            )
        params = torch.nn.ParameterDict(entries)
        if self.error_score_state is not None:
            self.error_score_state.resize(len(points))
        optimizers = {
            name: torch.optim.Adam(
                [{"params": [parameter], "lr": float(learning_rates[name]), "name": name}],
                eps=1e-15,
            )
            for name, parameter in params.items()
        }
        self.strategy.check_sanity(params, optimizers)
        return params, optimizers, self.strategy.initialize_state()

    def render(
        self,
        params: Any,
        sample: Any,
        *,
        with_range: bool,
        c2w_override: Any | None = None,
        active_sh_degree: int | None = None,
        background_rgb: Any | None = None,
    ) -> tuple[Any, Any, Any, dict[str, Any]]:
        torch = self.torch
        c2w = torch.as_tensor(
            sample.c2w if c2w_override is None else c2w_override,
            dtype=torch.float32,
            device=self.device,
        )[None]
        K = torch.as_tensor(sample.K, device=self.device)[None]
        camera_model = getattr(sample, "camera_model", "fisheye")
        if camera_model not in ("fisheye", "pinhole"):
            raise ValueError(f"unsupported sample camera_model {camera_model!r}")
        radial = (
            None
            if camera_model == "pinhole"
            else torch.as_tensor(sample.radial_coeffs, device=self.device)[None]
        )
        render, alpha, info = self.rasterization(
            means=params["means"],
            quats=params["quats"],
            # Parameters store log-scales (the convention the upstream MCMC
            # strategy operates on), but rasterization() expects LINEAR metric
            # scales - the upstream canonical call is torch.exp(splats["scales"]).
            # Passing the raw log values squared log(0.05)=-3 into the
            # covariance, rendering every 5 cm Gaussian as a ~3 m blob: real
            # scenes collapsed into structureless mush while the 2 m synthetic
            # fixture still "converged" and hid the bug.
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            **(
                {
                    "colors": torch.cat([params["sh0"], params["shN"]], dim=1),
                    # Progressive unlock: rasterization evaluates only the first
                    # (active+1)^2 bands, so early training cannot fake geometry
                    # with view-dependent color while poses/structure settle.
                    "sh_degree": self.sh_degree
                    if active_sh_degree is None
                    else min(self.sh_degree, max(0, int(active_sh_degree))),
                }
                if getattr(self, "color_model", "rgb_sigmoid") == "sh"
                else {"colors": torch.sigmoid(params["colors"])}
            ),
            viewmats=torch.linalg.inv(c2w),
            Ks=K,
            width=sample.width,
            height=sample.height,
            packed=False,
            render_mode="RGB-Ed" if with_range else "RGB",
            camera_model=camera_model,
            **({} if radial is None else {"radial_coeffs": radial}),
            # The KB4 fisheye needs the 3DGUT unscented path; pinhole faces
            # are linear so UT adds nothing, and turning it off unlocks
            # gsplat's Mip-Splatting antialiased compensation, which the
            # 3DGUT path explicitly rejects.
            with_ut=camera_model == "fisheye",
            with_eval3d=camera_model == "fisheye",
            global_z_order=False,
            rasterize_mode="classic"
            if camera_model == "fisheye"
            else getattr(self, "pinhole_rasterize_mode", "classic"),
        )
        rgb = render[0, ..., :3]
        if background_rgb is not None:
            # Composite un-saturated alpha onto an explicit background instead
            # of the implicit black canvas: with a bright overcast sky the
            # black bleed darkened 27% of the valid pixels by ~0.17 and the
            # whole frame by ~0.12. Depth stays un-composited.
            background = torch.as_tensor(
                background_rgb, dtype=rgb.dtype, device=rgb.device
            )
            rgb = rgb + (1.0 - alpha[0]) * background
        range_m = render[0, ..., 3] if with_range else None
        return rgb, range_m, alpha[0, ..., 0], info

    def strategy_post_step(
        self,
        params: Any,
        optimizers: dict[str, Any],
        state: Any,
        *,
        step: int,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        refine = (
            step < self.strategy.refine_stop_iter
            and step > self.strategy.refine_start_iter
            and step % self.strategy.refine_every == 0
        )
        before = (
            snapshot_gaussians(params, min_opacity=self.strategy.min_opacity)
            if refine
            else None
        )
        noise_scheduled = (
            self.strategy.noise_injection_stop_iter < 0
            or step < self.strategy.noise_injection_stop_iter
        )
        noise_probe_before = None
        noise_probe_observed = bool(
            state.get("_cloudstudio_noise_probe_observed", False)
        )
        if noise_scheduled and not refine and not noise_probe_observed:
            probe_count = min(len(params["means"]), 256)
            noise_probe_before = params["means"][:probe_count].detach().clone()
        self.strategy.step_post_backward(
            params=params,
            optimizers=optimizers,
            state=state,
            step=step,
            info=info,
            lr=optimizers["means"].param_groups[0]["lr"],
        )
        noise_position_delta_max_m = None
        if noise_probe_before is not None:
            probe_after = params["means"][: len(noise_probe_before)].detach()
            noise_position_delta_max_m = float(
                (probe_after - noise_probe_before).abs().max().cpu()
            )
            if noise_position_delta_max_m > 0.0:
                state["_cloudstudio_noise_probe_observed"] = True
        after = (
            snapshot_gaussians(params, min_opacity=self.strategy.min_opacity)
            if refine
            else None
        )
        return build_mcmc_step_event(
            step=step,
            before=before,
            after=after,
            refine_start_iter=self.strategy.refine_start_iter,
            refine_stop_iter=self.strategy.refine_stop_iter,
            refine_every=self.strategy.refine_every,
            noise_injection_stop_iter=self.strategy.noise_injection_stop_iter,
            noise_position_delta_max_m=noise_position_delta_max_m,
        )
