/*
 * MSVC host-compiler compatibility shim for gsplat's clean upstream checkout.
 *
 * gsplat f2d1413 uses __builtin_clzll in HOST code (ceil_log2_u64 in
 * RasterizeToPixelsFromWorld3DGSParallelBatchBwd.cu). MSVC has no such
 * builtin, which is why the first machine excluded the whole 3DGS kernel
 * group (BUILD_3DGUT=1) and thereby lost quat_scale_to_covar_preci_fwd /
 * full MCMC. Injecting this header via `nvcc -Xcompiler /FI<path>` maps the
 * builtin to the MSVC LZCNT intrinsic without modifying the locked source
 * tree, so the clean_vcs_commit provenance check still passes.
 *
 * Deliberately declares the intrinsic instead of including <intrin.h>:
 * force-including big system headers ahead of nvcc's own preincludes breaks
 * MSVC header self-consistency (C2953/C2011 redefinition storms).
 */
#pragma once
#if defined(_MSC_VER) && !defined(__builtin_clzll)
extern "C" unsigned __int64 __lzcnt64(unsigned __int64);
#define __builtin_clzll(x) ((int)__lzcnt64((unsigned __int64)(x)))
#endif
