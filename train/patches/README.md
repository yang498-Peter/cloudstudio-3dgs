# gsplat 本地补丁

`external/gsplat` 不入本仓库(见 .gitignore);对它的所有修改以补丁文件形式固化在这里,
重新克隆后用 `git apply` 重放。

## gsplat-s1-fisheye-keep-distortion.patch

- 基线 commit:`f2d14131483644e9977451b6403f6f0b73e6637f`(gsplat main, v1.5.3, 2026-07-02 克隆)
- 作用:colmap 数据路线支持**原始鱼眼直训**(3DGUT),不去畸变:
  1. `examples/datasets/colmap.py` Parser 增加 `keep_distortion` 开关:跳过鱼眼去畸变,
     改从 `<dataset>/masks/` 读圆形有效区 mask(tools/make_fisheye_masks.py 生成,
     255=有效),按 factor 缩放后进 `mask_dict` → 训练 loss 自动只算 mask 内像素
     (simple_trainer 原生行为)。
  2. `examples/simple_trainer.py`:环境变量 `S1_KEEP_FISHEYE=1` 时启用上述开关,
     并把每个相机的 k1–k4 以 ncore 同款鸭子类型表(`self.ncore_camera_data`)喂给
     UT 光栅化路径(`camera_model="fisheye"` + `radial_coeffs`),渲染代码零改动。

重放:

```powershell
cd external\gsplat
git apply ..\..\train\patches\gsplat-s1-fisheye-keep-distortion.patch
```

升级 gsplat 版本时:先 `git stash`/重放补丁,冲突则按上述两点语义手工移植,
并重新生成补丁文件 + 更新本 README 的基线 commit。
