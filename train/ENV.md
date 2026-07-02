# 训练环境搭建(Windows + RTX 5070 Laptop 记录,2026-07-02)

## 前提(本机已具备)

- NVIDIA 驱动 592.01(CUDA 13.1 driver)+ **CUDA Toolkit 12.8(nvcc)**
- **PyTorch 2.11.0+cu128**(Blackwell/sm_120 需要 cu128 系)
- VS2022 BuildTools(MSVC x64)
- Python 3.12(全局环境)

## gsplat 安装(源码 + CUDA 编译)

```cmd
git clone --depth 1 https://github.com/nerfstudio-project/gsplat.git external\gsplat
cd external\gsplat
git submodule update --init --depth 1 gsplat/cuda/csrc/third_party/glm
git apply ..\..\train\patches\gsplat-s1-fisheye-keep-distortion.patch

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set VSLANG=1033
set DISTUTILS_USE_SDK=1
pip install -e . --no-build-isolation
```

### 踩过的坑(按遇到顺序)

1. **中文版 MSVC + torch 编译探测**:`cl.exe` 中文横幅让 torch `_check_abi` 的 OEM 解码崩
   (`UnicodeDecodeError ... 'oem' codec`)→ `set VSLANG=1033` 强制英文输出。
2. **`DISTUTILS_USE_SDK`**:VC 环境激活后 torch 要求 `set DISTUTILS_USE_SDK=1`,否则直接 raise。
3. **glm 缺失**(`fatal error C1083: glm/glm.hpp`):gsplat 用 submodule 带 glm,
   `--depth 1` 克隆不含;只需拉 glm 这一个 submodule(googletest 不用,网络慢时省时间)。
4. **MSVC 不认 `__builtin_clzll`**(`error C3861`,ParallelBatchBwd.cu 主机函数):
   源码修复见 train/patches(`_BitScanReverse64` 分支)。
5. **编译 >60 分钟的主因:默认编全架构**。`set TORCH_CUDA_ARCH_LIST=12.0`
   (RTX 5070 桌面/笔记本都是 Blackwell sm_120)+ `set MAX_JOBS=%NUMBER_OF_PROCESSORS%`
   后预计 ~10 分钟。已写进 `setup_new_machine.cmd`。
   (在旧机器上从未编译成功过——编到一半为换机主动中止,新机器是首次完整编译。)

## examples 依赖 —— 不要整包装 requirements.txt!

`examples/requirements.txt` **钉死 `torch==2.9.1` + `torchvision==0.24.1`**,
照装会把 cu128 的 torch 降级成公版,Blackwell 直接废。手工挑选安装:

```cmd
pip install viser nerfview pycolmap torchmetrics tyro pyyaml tensorboard ^
  imageio[ffmpeg] opencv-python-headless scipy scikit-learn matplotlib tqdm ^
  piexif splines tensorly
pip install torchvision --index-url https://download.pytorch.org/whl/cu128 --no-deps
```

跳过的可选依赖(需要时再装):
- `nvidia-ncore` — NVIDIA 多传感器数据 SDK,只有 `data_type=ncore` 用;我们走 colmap 路线。
- `fused-ssim` / `fused-bilagrid` / `ppisp`(git+CUDA 编译)— gsplat.losses 有纯 torch 回退;
  PPISP 曝光补偿是 Phase 2 计划项,启用时再装 `ppisp @ git+https://github.com/nv-tlabs/ppisp@v1.0.0`。

## 冒烟训练

`train\run_smoke_gs2.cmd`:3DGUT(`--with_ut --with_eval3d`,原始鱼眼 + 圆形 mask,
需 `S1_KEEP_FISHEYE=1`)+ MCMC(cap-max 1M)+ factor 4(728px),10k 步。
8GB 显存的经验值;正式训练(2912 全分辨率、更多高斯)需 ≥24GB 卡。
