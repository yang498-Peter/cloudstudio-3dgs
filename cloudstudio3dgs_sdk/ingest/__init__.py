"""Dataset ingestion for the CloudStudio 3DGS SDK.

``adapters`` turn a capture on disk into a :class:`~.bundle.DatasetBundle`;
``caches`` turns a bundle plus a profile into the ordered list of signed caches
the training recipe consumes; ``tiling`` derives tile boxes for scenes nobody
hand-cut.

    from cloudstudio3dgs_sdk.ingest import load_dataset, plan_caches
    bundle = load_dataset(r"C:\\Peter\\testdata\\S1\\house0305")
    plan = plan_caches(bundle, profile)
    print("\\n".join(plan.build(dry_run=True)))
"""

from __future__ import annotations

from .adapters import ADAPTERS, adapter_by_name, detect_adapter, load_dataset
from .bundle import (
    BundleImage,
    CameraIntrinsics,
    DatasetBundle,
    PointCloudRef,
    RigTransform,
    describe_capabilities,
    load_bundle_manifest,
    verify_bundle_manifest,
    write_bundle_manifest,
)
from .caches import CachePlan, CacheProfile, CacheSpec, build_cache_specs, plan_caches
from .errors import (
    BundleSignatureError,
    DatasetDetectionError,
    DatasetIncompleteError,
    GpuStepRequired,
    IngestError,
)
from .tiling import (
    SlabPlan,
    TilingRule,
    build_slab_tile_plan,
    histogram_from_las,
    histogram_from_points,
    slab_split,
)

__all__ = [
    "ADAPTERS",
    "BundleImage",
    "BundleSignatureError",
    "CachePlan",
    "CacheProfile",
    "CacheSpec",
    "CameraIntrinsics",
    "DatasetBundle",
    "DatasetDetectionError",
    "DatasetIncompleteError",
    "GpuStepRequired",
    "IngestError",
    "PointCloudRef",
    "RigTransform",
    "SlabPlan",
    "TilingRule",
    "adapter_by_name",
    "build_cache_specs",
    "build_slab_tile_plan",
    "describe_capabilities",
    "detect_adapter",
    "histogram_from_las",
    "histogram_from_points",
    "load_bundle_manifest",
    "load_dataset",
    "plan_caches",
    "slab_split",
    "verify_bundle_manifest",
    "write_bundle_manifest",
]
