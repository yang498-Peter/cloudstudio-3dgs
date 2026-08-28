# snow MipMap 对位路线（2026-08-27）

本目录固化本轮最终晋级配置。完整证据和失败实验见
`docs/2026-08-27_MipMap_AT与Face4逆向分析.zh-CN.md`。

## 输入与阶段

1. 使用 `at_pose_only_v22c` 生成的签名训练 Manifest；
2. 主表面为 `v22e@2000` 的 6,158,096 个 SH0 高斯；
3. 追加 100,000 个确定性远场 sky 高斯；
4. 运行 `v22j_fullres_frozen_sky_polish.config.json`：2912×2912 原始鱼眼、500 步、冻结 means/scales/quats，只优化 SH0 颜色、不透明度和曝光；
5. 按签名 sky 边界分别导出 combined、surface 和 sky PLY。

## 复现命令

```powershell
python tools\augment_checkpoint_sky.py `
  --checkpoint outputs\snow-20260224-full-20260825\training_independent_at_sh0_fullres_formal2000_v22e\checkpoints\best_golden.pt `
  --dataset-manifest outputs\snow-20260224-full-20260825\independent_at_training_manifest_v22c\dataset_manifest.json `
  --output outputs\snow-20260224-full-20260825\sky_augmented_v22i\warm_start_with_sky.pt `
  --report outputs\snow-20260224-full-20260825\sky_augmented_v22i\sky_layer_report.json `
  --count 100000 --radius-m 136.5 --scale-m 1.1 --opacity 0.05 `
  --rgb 0.68 0.75 0.93 --min-world-z-direction -0.4

python tools\train_gsplat.py `
  --config experiments\snow_mipmap_route_20260827\v22j_fullres_frozen_sky_polish.config.json
```

最终 run Manifest SHA256 为
`7ca813bd3cc57c6892693676ae2d64d26c72e2c56a23a9a0d0df74ab05f5e973`；质量报告状态为
`COMPLETE`，SHA256 为
`84b2b82162747d573ac6171bef9b171c2df1e14baa72411a8bd6939fab906005`。
