# MVP S1 扫描仪数据格式规格(Phase 0 交付物)

> 基于真实数据核验,2026-07-02。样本:
> `D:\S1\PROCESS\2026-01-29_14-36-28-Finland`(完整管线)与
> `G:\S1\2026-06-17_12-40-48gs2\process\2026-06-17_12-40-48gs2_3`(解算 run)。
> CloudStudio 侧的参考实现:`cloudstudio-windows/web-uploader/lib/mvps1-solver.js`
> 与 `web-uploader/scripts/process_s1_panoramas.py`(KB 鱼眼投影既有代码)。

## 1. 原始记录目录结构(扫描仪落盘)

```
<recording>/                       例: 2026-06-17_12-40-48gs2/
├── metadata.yaml                  # 会话信息(时长、image_num、rtk_fixed_ratio、ctrl_point_num)
├── camera/
│   ├── left/<ns-timestamp>.jpg    # 左鱼眼, 2912x2912, ~2.5MB/张, 文件名=纳秒时间戳
│   └── right/<ns-timestamp>.jpg   # 右鱼眼, 同上
├── info/
│   ├── calibration.json           # ★ 相机标定(见 §2)
│   └── device_info.json           # 设备型号 MVP-S1 / 固件 / SN
├── data/data_raw.mcap             # 全传感器流(LiDAR+IMU+相机触发), 3–8 GB
├── odom-realtime.csv              # 实时 SLAM 轨迹(格式同 §4 odom.csv)
├── colorized-realtime.las         # 实时着色点云(预览级)
├── Preview.pcd                    # 快速预览点云
└── process/<runName>/             # ★ metacam_cli 解算输出(见 §3), 可多个 run
```

CloudStudio 侧的输入判定标记(`MVP_S1_RAW_INPUT_MARKERS`, mvps1-solver.js:54-62):
`metadata.yaml|yml`、`data_raw.mcap|data.mcap`、`odom-realtime.csv`、`colorized-realtime.las|laz`。

## 2. 相机标定 `info/calibration.json`

真实样本(Finland, 标定时间 2026-01-04):

```jsonc
{
  "calibration_time": "2026-01-04_15-39-17",
  "version": "v1",
  "cameras": [
    {
      "name": "left",                 // left / right 两目
      "type": "fisheye",
      "width": 2912, "height": 2912,
      "intrinsic": { "fl_x": 788.665, "fl_y": 788.779, "cx": 1451.269, "cy": 1449.111 },
      "distortion": {
        "camera_model": "OPENCV_FISHEYE",   // = Kannala-Brandt 4 参数
        "params": { "k1": 0.0808, "k2": -0.0091, "k3": -0.0052, "k4": 0.00038 }
      },
      "transform_from_lidar": {       // 相机-LiDAR 外参
        "rotation": [[3x3]],
        "position": [x, y, z]         // 米, 量级 ~3-7cm
      }
    },
    { "name": "right", ... }
  ],
  "imu": [ { "name": "internal", "lidar_to_imu_transform": { rotation, position } } ]
}
```

结论(对应研究计划附录待确认项):
- **畸变模型 = OPENCV_FISHEYE(KB4 等距投影 + k1–k4)** —— 3DGUT/Fisheye-GS 均直接支持。
- **FoV 估算**:边缘半径 ≈ cx ≈ 1451px,f ≈ 788px → theta_d ≈ 1.84 rad,
  反解畸变后全 FoV 约 **190°–200°**(不同批次镜头 f 在 777–789 波动,内参逐台标定)。
  → 研究计划 §3.2 的 160° 裁剪/mask 问题**成立,Phase 1 需做消融**。
- 左右目内参独立标定,注意逐台设备/逐次标定值不同,**必须从每条记录自己的
  calibration.json 读取,不可硬编码**。

## 3. 解算输出 `process/<runName>/`(metacam_cli,即 mvps1 solver)

```
process/<runName>/
├── transforms.json        # ★ NeRF 风格关键帧位姿(见 §5) —— 3DGS 首选输入
├── ImgPose.txt            # ★ 全部图像位姿, 局部坐标系(见 §4)
├── odom.csv               # 优化后轨迹(LiDAR 频率, ~10Hz)
├── colorized.las          # ★ 着色点云, 局部坐标系(~1GB/场景)
├── uncolorized.las        # 未着色点云(同几何)
├── undistort/left|right/  # 去畸变图(90° FoV 针孔, 1600x1600, f=800) —— 备用路线
├── converted/             # Potree 格式(CloudStudio 查看用, 3DGS 不用)
├── SCANS/                 # 中间 .pcd(CloudStudio 成功后会清掉, 不依赖)
├── tmp.json / task.log    # 进度与日志
└── geo/                   # (若有 RTK/GCP) ImgPose_geo.txt, ecef_ImgPose.txt, ecef_colorized.las
```

Finland 样本根目录另有 `all_frames.json`(全帧版 transforms)、`params.json`
(解算参数,含 `enablePano`、`humanImageErase` 等开关)、`QualityReport_*.pdf`。

### 坐标系红线

3DGS 训练**只用局部坐标系**产物:`ImgPose.txt` / `transforms.json` / `colorized.las`
(首帧在原点附近,数值量级安全)。**禁止**使用 `geo/ecef_*`(ECEF 坐标 ~6.4e6 米,
float32 精度直接崩;CloudStudio 有 scanner-ecef-guard.js 挡同类问题,本工程同理)。
若客户要求成果落地理坐标,训练完在**导出阶段**对 PLY 整体施加 `geo_info.csv` /
ImgPose_geo 派生的相似变换。

## 4. 位姿文本格式

### `ImgPose.txt`(逐张图像,空格分隔,一行表头)

```
index x y z roll pitch yaw qx qy qz qw timestamp
left/1781696580405763840.jpg 0.000749 0.014339 -0.026333 92.428 -179.555 154.633 -0.675690 0.149193 -0.161129 0.703720 1781696580.405763864517
```

- `index` = 相对 `camera/` 的图像路径(left/... 与 right/... 都有)。
- 平移单位米,角度为度,四元数 `qx qy qz qw`;时间戳秒(小数)。
- **约定(已实测钉死,2026-07-02)**:`x y z` = 相机中心在世界系的位置;
  四元数 = **camera-to-world 旋转,OpenCV 相机轴(x右 y下 z前)**。
  验证方法:与同帧 transforms.json 矩阵数值对比,`R_quat = R_tf @ diag(1,-1,-1)`
  残差 1.5e-5(house2 数据,208 个共同帧)。`roll/pitch/yaw` 列冗余,不要用。
  (mipmap 适配层的 `imgpose_world_to_camera + transpose` 说法与此一致:
  它的 SDK 要 w2c,所以对 c2w 旋转做转置。)

### `odom.csv`(LiDAR 频率轨迹,~10Hz)

```
#timestamp, x, y, z, q_x, q_y, q_z, q_w, pts, rtk_ppk_fix, ctrl_pt_idx
1765818809300024986, -0.00974, -0.00596, 0.00422, 0.00357, 0.17886, -0.00277, 0.98386, 0, 0, 0
```

时间戳纳秒整数。3DGS 一般不直接用它(ImgPose 已是插值到图像时刻的位姿),
但可用于轨迹可视化与"离轨迹 1–2m 新视角"压力测试的机位生成。

## 5. `transforms.json`(★ 3DGS 首选输入)

解算器直接输出的 NeRF 风格文件,顶层三个键:

```jsonc
{
  "frames": [                      // 关键帧(左右目各半; gs2 样本 174 帧 = 87L + 87R,
    {                              //  从 1246 张原图按阈值筛选)
      "file_path": "left\\1781696580405763840.jpg",   // 注意反斜杠, 相对 camera/
      "timestamp": 1781696580405763840,
      "w": 2912, "h": 2912,
      "fl_x": 777.479, "fl_y": 777.752, "cx": 1453.937, "cy": 1450.787,
      "k1": 0.0818, "k2": -0.0119, "k3": -0.0029, "k4": -0.00012,  // 逐帧带 KB4 内参
      "transform_matrix": [[4x4]]  // ★ camera-to-world, OpenGL/nerfstudio 轴向(已实测钉死)
    }, ...
  ],
  "metainfo": {                    // 关键帧筛选参数
    "merge_mode": "average", "threshold_angle": 15,
    "threshold_distance": 2, "threshold_motion": 0, "version": "3.8.7"
  },
  "undistort_camera_model": {      // undistort/ 文件夹对应的针孔模型
    "fov_degree": 90, "width": 1600, "height": 1600,
    "intrinsic": [[800,0,800],[0,800,800],[0,0,1]]
  }
}
```

含义与用法:
- **这就是"免 SfM"的落地形态**:每张关键帧图已带位姿 + 鱼眼内参,格式与
  nerfstudio `transforms.json` 高度同源(差异仅剩:`file_path` 反斜杠、无
  `camera_model` 顶层字段)。转换器只需做轻量归一化。
- **位姿约定(已实测钉死,2026-07-02)**:`transform_matrix` = **camera-to-world,
  OpenGL/nerfstudio 相机轴(x右 y上 z朝后)** —— 即 nerfstudio 原生约定,可直接喂。
  验证:`tools/reproject_check.py` 四约定叠加图目视比对,左右目各 2 帧,
  `c2w_gl` 全部严丝合缝(见 `experiments/reproject_gs2/`);且与 mipmap 适配层
  生成的 SDK orientation 的数值关系(`R_sdk = R_tf @ diag(1,-1,-1)`)互证。
- 关键帧密度受 `threshold_distance=2m / angle=15°` 限制 —— 对 3DGS 训练可能偏稀,
  需要更密时改用 `ImgPose.txt`(全图像)+ `calibration.json` 自行组帧,
  或 Finland 样本的 `all_frames.json`。
- `undistort/`(90° 针孔)是研究计划 §3.2 的"路线 A"现成产物:损失视场但零投影风险,
  可作为 Phase 1 的对照组/兜底路线。

## 6. 与 CloudStudio 处理流程的对应关系(只读参考,不复用代码)

| CloudStudio 环节 | 位置 | 对本工程的意义 |
|---|---|---|
| 原始数据校验 | mvps1-solver.js `assertMvpS1RawInputDirectory()` | inspect_recording.py 的判定标准来源 |
| KB4 鱼眼投影实现 | web-uploader/scripts/process_s1_panoramas.py:125-167 | reproject_check.py 的投影公式参照 |
| 解算调度 | mvps1-solver.js → metacam_cli.exe(8 阶段) | 本工程消费其输出,不重跑 |
| ECEF 防护 | web-uploader/lib/scanner-ecef-guard.js | §3 坐标系红线的依据 |

## 7. mipmap(竞品)SDK 的数据消费方式(对标参考)

详见 `docs/references/MVP_S1_3DGS输入数据与SDK调用整理_20260702.zh-CN.md`
(对成品软件 `G:\S1\3DGS\Tersus-GNSS-MVP-S1-3DGS-Processor-V1.0.0-alpha2-20260608B` 的逆向整理)。
要点:

- 它消费的输入与本规格一致:双鱼眼图 + `ImgPose.txt`(优先)/`transforms.json`(备用)
  + `colorized.las`,**不用** `data_raw.mcap`,默认也不用 odom 轨迹。
- 图像使用策略:左右按时间戳配对(容差 50ms),**均匀采样 935 对 ≈ 1870 张**入训练。
  → 自研的采样消融基线。位置先验不确定度 `pos_sigma=[0.03,0.03,0.06]`,
  说明其 SDK 内部会在此先验下**微调位姿**(gsplat pose refinement 对应项)。
- 新批次数据注意:calibration.json 已有 `"version": "v2"`;process 目录的 LAS 带项目
  前缀(`<project>_colorized.las`,CloudStudio 友好命名);LAS 为 1.4/PF7,
  其 SDK 需转 1.2/PF3 —— 自研管线用 laspy 直读 1.4/PF7,无此包袱。
- 它有 `remove_moving_object: true` 与 sky.ply 分离输出 → 动态剔除与天空处理
  **不是**我们独有卖点,是必须追平的基线;超越点在 LiDAR 交叉验证与几何精度。
- 输出:`gs.ply`(2.5GB/4855 万高斯)+ `ue/gs_full.ply`(11.2GB)+ SOG tiles
  → Phase 4 的压缩/分块/LOD 必须从一开始设计。

## 8. 剩余待确认项(研究计划附录)

- [ ] 快门类型(全局/卷帘):device_info.json 未声明,需问硬件团队;若卷帘,3DGUT 喂行曝光时间。
- [x] 鱼眼畸变模型:OPENCV_FISHEYE / KB4。
- [x] POS 格式与轴向约定:transforms.json = c2w/OpenGL;ImgPose.txt = c2w/OpenCV(§4/§5)。
- [x] 标定+位姿质量初验:reproject_check 目视全部贴合(gs2 场景,左右目各 2 帧),
      Phase 1 重投影门槛**初步通过**;正式 Gate 前再对 2–3 个场景重复并做逐点误差统计。
- [x] 场景类型:样本覆盖室内(house)、室内外混合(ATS-IN-OUT)、室外(RTK/snow),见 DATASETS.md。
- [x] 本机 GPU:RTX 5070 Laptop 8GB(冒烟可用;正式训练建议 ≥24GB 或云端)。
- [x] mipmap 对标:成品软件 `G:\S1\3DGS\Tersus-GNSS-MVP-S1-3DGS-Processor-V1.0.0-alpha2-20260608B`,
      成功输出 `G:\Tersus3DGSResults\20260609-234131-s1-3dgs\`,数据契约见 §7 / references 文档。
