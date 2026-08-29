# snow MipMap 路线交叉核对与重训决策

日期：2026-08-28（Asia/Singapore）

## 1. 结论

V30/V31 不是质量冠军，当前不得继续堆训练步数，也不得直接复用 V26a 或 MCMC 长训。下一轮必须先完成一个单 Tile、502 step 的证据对位边界；未通过前不启动全域重训。

本轮交叉核对确认，过去把三类东西混在了一起：

1. MipMap snow 实际执行的算法；
2. DLL 中存在但 snow 未启用的能力；
3. CloudStudio 为 LiDAR 场景增加的安全增强。

后续配置和门禁必须显式区分三者。

## 2. 三路证据

### 2.1 实际任务与最终产物

- 竞品任务根为 `D:\mipmap-lite\1fad9647-f717-4cd4-9391-11f219d7e5d1\snow\snow-20260827`。
- `info.json` 直接记录 `type=lidar`、`resolution_level=1`、`remove_moving_object=true`，任务正常完成。
- 最终 surface `gs.ply` 为 `6,018,902` 个 Gaussian。
- 四个 Tile level-0 数量为 `1,773,436 / 1,988,450 / 1,506,518 / 750,498`，相加恰好为 `6,018,902`。
- 最终 surface PLY 可按 `Tile_0 → Tile_1 → Tile_2 → Tile_3` 分段逐属性匹配四个 level-0 产物，说明 snow 结果是四块完整数据的顺序连接，没有跨块参数平均或去重。
- `sky.ply` 与 `sky_full.ply` 各有 `100,000` 个独立 sky Gaussian；主 surface PLY 的点数不包含 sky。
- `sky_full.ply` 前 9 个 SH-rest float 在约 `99.365%` 的 sky Gaussian 上非零，剩余 36 个为零，直接证明天空使用 degree-1；surface 的 45 个 SH-rest 在抽样中全部为零。
- 本轮重新读取磁盘并复验 SHA256：surface `gs.ply` 为 `C10026FEFF1DD645273E1620C7BA7E1A08C8727282734D847CC1CBA3D81DF8C4`，`sky.ply` 为 `C55F326E99AF6F58C3A8E00B63687CAFCEAFBFE20984C710B6EE05D8D2CDA3E5`，`sky_full.ply` 为 `3969AA5491907D66D3CD04727C3D898412FB8F40E20EA6D223756331D0F3DD29`。四个 level-0 文件也均满足 `4 + 56×count` 的精确长度关系。

### 2.2 二进制静态审计

- snow/type-2 选择 `AfterTrain`，没有选择 `AfterTrainMCMC`；MCMC flag 和 redundancy-cull flag 均为 false。
- Gaussian 生命周期从 step 500 开始，每 100 step 执行；候选要求二维累计梯度大于 `1.5e-4` 且 opacity 大于 `0.15`。
- 小尺度候选 clone，大尺度候选 split；split 生成两个子点，尺度除以 `1.6`。
- cull 阈值为前半程 opacity `0.10`、后半程 `0.05`，world scale `0.2`、screen radius `0.15`；每 300 step 把 opacity 上限 clamp 到 `0.2`。
- High/type-2 使用 `[5,10,5]` 个完整 view epoch，每个 epoch 重新做无放回 Fisher–Yates 置换，总步数为 `20×V`。
- 初始化为 LiDAR Tile 点云：K7（自身加 6 邻点）尺度、`[d,d,0.5d]` 三轴、K30 局部法向、+Z 到法向的最短弧四元数、RGB→SH0、opacity 初值 `0.1`。
- 六组 Adam 学习率为 xyz `1.6e-5→1.6e-6`、scale `0.005`、rotation `0.001`、SH-DC `0.0025`、SH-rest `0.000125`、opacity `0.05`。
- 照片项为 `0.6 mean-L1 + 0.4(1-SSIM)`；另有 DA2 `0.5`、mesh depth `0.5→0.25`、mesh normal `0.05`、后期自洽 normal `0.01`、opacity mean `0.01`、后半程条件 sky opacity `0.04`。
- per-camera BilateralGrid 已启用，张量为 `[N_camera,12,8,16,16]`，LR `0.002`、TV 权重 `5.0`；SIFT pose refine preset 也已开启并受运行质量门限控制。
- renderer mask 的可确认规则是 `(seg!=255)&(seg!=33)`。

### 2.3 我方失败实验

- V26a 虽复刻了主要生命周期阈值，但只使用 SH0、LiDAR range `0.05` 和简化曝光模型；没有竞品 BilateralGrid、完整 mesh depth/normal、DA2、真正 SegFormer label 33 语义和 SIFT refine。
- V26a 还加入了 AbsGrad、revised opacity、LiDAR 切平面出生守卫、额外 scale/anisotropy 正则，并在 `15V` 停止生长。这些是 CloudStudio 增强或实验，不是 snow 竞品原样参数。
- V26a Tile_1 从 `971,903` 个点训练到 step 2618 只剩 `360,078` 个，出现高轴比薄针状高斯；它不是竞品行为复现。
- MCMC step 5236 虽降低照片 loss，却使最大尺度增至 `1.433 m`、normal error 和 dead opacity 明显恶化；MCMC 不是 snow 竞品主路线。
- V30/V31 把 100k SH0 照片球壳和 surface 合成单一 PLY。前景 LiDAR 像素 alpha 均值仅 `0.7134`、P05 `0.0429`，近半前景低于 `0.9`；加入天空后 foreground RGB 平均改变量超过 `0.10`，直接造成墙体透天和背景串色。

## 3. 已确认、增强、未知三分法

| 类别 | 内容 |
|---|---|
| 竞品已确认 | 四 Tile 串行；LiDAR K7/K30 初始化；20V 无放回视图；经典 gradient clone/split/cull/reset；六组 Adam；RGB/双 depth/双 normal/opacity 调度；BilateralGrid；条件 SIFT；独立 100k SH1 sky；surface/sky 分开交付 |
| CloudStudio 增强 | 新生点 LiDAR planarity/support gate；切平面出生；不支持出生拒绝；唯一 core ownership；DA2/mesh 可关闭的 LiDAR-first 基线；逐 checkpoint alpha/泄漏/几何门 |
| 仍未知 | label 33 类名和模型阈值；surface SH-rest 是否真正训练；cap 公式中的厂商倍率 C；每次 lifecycle 的真实 clone/split/cull 数；TrainBackground 的完整 loss、初始化与 opacity 调度 |

## 4. 路线调整

### 4.1 立即禁止

- 不从 V31 继续 refine。
- 不再把 sky append 到 surface PLY。
- 不用 MCMC 作为第一主线。
- 不把 V26a 的同名阈值称为竞品等价实现。
- 不以整图平均 PSNR/SSIM 单独晋级。

### 4.2 下一次单 Tile 边界

只准备一个新的 Tile_1 502-step 边界，不启动长训。边界配置必须逐项声明来源：

1. 竞品确认项：K7/K30 初始化、opacity 0.1、六组 Adam、20V 置换调度、RGB 0.6/0.4、step 500/100 生命周期。
2. CloudStudio 增强项：LiDAR 新生支持与切平面 proposal；必须单独记账，不写成竞品参数。
3. 暂缓项：DA2 和完整 mesh 继续关闭，直到 LiDAR-only 基线证明不足；关闭会改变 opacity/cull 的统计分布，因此不能照搬竞品 opacity cull 后直接宣称行为等价。
4. 先补项：真实 renderer semantic/dynamic mask 消费、per-camera BilateralGrid 或等价受控光度模型、surface/sky 分离交付。
5. surface 表示先以最终产品可证实的 DC-only 为基线；如做 SH1，只能作为独立 A/B，不能标成已证实竞品 surface 行为。

### 4.3 502-step 通过条件

- step 500 前后 clone/split/cull 数和总点数可追踪；不得出现 V26a 式无解释大规模净删除。
- 初始 LiDAR Gaussian 与 newborn 分开统计 opacity、scale、支持率和 cull 原因。
- 墙体/地面 foreground alpha 不下降；天空关闭时不允许靠白背景掩盖透明。
- Gaussian 最大尺度、P95/P99、轴比和点到 LiDAR 表面距离不劣于 A0 基线。
- 固定原鱼眼 ROI 同时检查墙、远树、雪面、人物/动态区；未通过任一 ROI 即停止。
- 只有上述边界全部 PASS，才允许继续到下一完整 epoch；仍不得直接启动五 Tile 或全域长训。

## 5. 当前状态

当前状态为 `COMPETITOR_CROSSCHECK_COMPLETE_RETRAIN_BLOCKED`。没有启动新训练。下一项实现工作是修正 surface/sky 交付合同，并补齐 renderer semantic mask 与光度 nuisance model 的消费证据，然后再生成新的单 Tile 签名边界配置。

## 6. V33a 真实边界与连续生命周期复核（2026-08-29）

V33a 已把主线切为 `projected-gradient Clone/Split/Cull + LiDAR tangent birth guard`，并加入逐相机 PPISP 光度模型；MCMC、DA2、POS 优化均关闭，surface 保持 SH0。真实 Tile_1 全分辨率 step 502 边界通过：初始 `971,903` 个高斯，clone `244,484`、split 父点 `5`、cull `282,522`，事件后为 `933,870`，保留率 `96.086%`。候选父点和新生子点 LiDAR 支持均值分别为 `95.726% / 95.280%`，点到 LiDAR 最近距离 P95 为 `4.837 mm`，没有高斯超过 `0.3 m`，最长轴不超过 `0.2 m`。这证明 projected-gradient 出生、LiDAR 父点门和切平面 proposal 本身已经连通且几何健康。

同一签名状态继续到 step 1002 后发生明确 population collapse：六次生命周期累计 clone `857,210`、split 子点净增 `484`、cull `1,361,394`，最终只剩 `468,203`，相当于初始点数的 `48.17%`。其中 step 600 执行 opacity reset，将全部高 opacity 压到 `0.2`；step 700 随即 cull `361,844`，占事件前点数 `43.47%`。该事件的低 opacity 候选为 `359,585`，与 cull 总数几乎相同，因此失败主因不是 LiDAR 出生守卫，也不是 clone 不足，而是当前 `reset_every=300`、`reset_opacity_cap=0.2` 与前段 `prune_opa=0.1` 在我方视图/损失/可见性统计下形成破坏性组合。

step 502 原鱼眼验证为 PSNR `6.9918 dB`、SSIM `0.45841`、alpha 均值 `0.17971`；相对固定拓扑 A0 step 2618 的 `6.9532 / 0.46069 / 0.17727` 没有出现首次生命周期后的立即崩坏，但也尚未证明画质晋级。step 502 完整 SH0 PLY 为 `training_tile1_v33a_ppisp_lidar_boundary502/exports/snow_tile1_v33a_step502_sh0_full.ply`，包含 `933,870` 个高斯。step 1002 结果判定失败，不导出为候选产品。

路线决策因此更新为：MCMC 正式退出当前主训练 backbone，仅保留为未来固定预算的 Surface Budget Relocation 研究；固定拓扑 A0 保留为几何和覆盖基线；正式研发主线为 `LiDAR planar surfel + projected-gradient adaptive topology + LiDAR-safe birth + observation/coverage-aware cull`。不再直接复用当前 opacity cull/reset 组合，也不启动五 Tile 长训。

下一轮只做 Tile_1 短边界：先增加 opacity、world-scale、screen-radius 三类 cull 原因的独立遥测，再让低 opacity 点同时满足“重置后获得足够有效观测、连续至少两个生命周期低于阈值、局部表面仍有替代覆盖”才允许删除；world/screen 超限仍可即时剔除。验收要求包括单次净删除不超过事件前 `5%`、step 1002 总保留率至少 `90%`、LiDAR 支持率至少 `90%`、点到面 P95 不超过 `1 cm`、前景 alpha 不低于 A0，以及墙体、远树、雪面和动态区四类 ROI 均不退化。

当前状态更新为 `V33A_BOUNDARY_PASS_CONTINUED_CULL_FAILED`。step 502 可作为诊断 checkpoint；step 1002 不得晋级或继续长训。

## 7. V34a observation-aware Cull 修复与短跑结果（2026-08-29）

针对 V33a 的 collapse，V34a 保留竞品确认的 projected-gradient、Clone/Split、`0.10/0.05` opacity 阈值、`0.2 m` world-scale、`0.15` screen-radius 与每 300 步 opacity reset，只对 opacity 删除增加 CloudStudio 安全门：至少 4 次有效观测、连续两个生命周期低于阈值、reset 后 200 步宽限期，以及每次最多删除当前人口的 5%。world-scale 和 screen-radius 异常不受宽限，仍即时删除。生命周期遥测同时拆分 raw opacity、实际 opacity、world scale、screen scale 和重叠数量。

真实 Tile_1 step 502 从 `971,903` 个 LiDAR 高斯开始：clone `244,483`、split 父点 `5`、opacity cull `0`、world/screen 几何 cull 合计 `1,858`，事件后为 `1,214,533`。raw 低 opacity 候选虽有 `280,707`，但因尚未连续两轮而未被误删。新生点 proposal 应用率 `100%`，父/子 support mean 为 `95.726% / 95.281%`，峰值显存约 `2.47 GiB`，边界报告状态为 PASS。

同一签名状态继续到 step 1002 后为 `1,573,987` 个高斯，而失败 V33a 同阶段只有 `468,203`。六次生命周期累计生成 clone/split children `761,663`，cull `159,060`，另有 `519` 个 split parent 被两个子点替换，净增 `602,084`；step 600 和 900 的 opacity cull 分别为 `66,745 / 77,931`，均被 5% 上限约束。step 700、800、1000 处于 reset 宽限期，opacity cull 为 0，仅删除尺度/屏幕异常点。没有再次发生单轮 40% 以上删点。

几何审计显示：可见 Gaussian 到 LiDAR 最近距离 P95 `9.023 mm`、P99 `18.68 mm`，超过 `0.3 m` 为 0；最长轴 P50/P95 为 `7.90/26.40 mm`，没有超过 `0.5 m` 的巨型点。可见高斯轴比 P50 为 `5.68`，高于初始化约 2，说明已开始形成薄盘，但仍低于竞品终态约 12.54。opacity 方面仍有 `235,980` 个 `<0.005` 死点和 `775,382` 个 `0.005–0.1` 雾点，证明保护覆盖的同时积累了较大低贡献人口，尚未建立最终 population equilibrium。

原鱼眼 36 帧验证为 PSNR `6.9944 dB`、SSIM `0.45452`、alpha 均值 `0.18801`、LiDAR 区域 alpha `0.27570`。相对 V33a step 502，alpha 从 `0.17971` 提升、LiDAR alpha 从 `0.25638` 提升，PSNR 基本持平，但 SSIM 从 `0.45841` 下降。因此当前只能证明覆盖与生命周期稳定性改善，不能宣称清晰度晋级。step 1002 完整 SH0 PLY 为 `training_tile1_v34a_ppisp_lidar_cullsafe_review1002/exports/snow_tile1_v34a_step1002_sh0_full.ply`，包含全部 `1,573,987` 个高斯，SHA256 为 `D13153718115FA90D16441385A20AE22DECF78590566DB4468DC05B5308FE26E`。

同时修正共享遥测口径：经典 3DGS 事件过去会把 refine 前的低 opacity 数量误写到通用 `relocated_count`，造成 MCMC 关闭时仍显示数百万 relocation。新代码以 `classic_lifecycle` 的 clone、split、cull 为权威，经典路线的 relocation 固定为 0，并在 resume 时修复已加载的旧统计；该修复只改变证据口径，不重写已有 checkpoint 或模型参数。

当前状态为 `V34A_CULL_COLLAPSE_FIXED_QUALITY_GATE_PENDING`。不得直接进入 2618 或五 Tile 长训。下一步先由相同 SuperSplat 视角比较 A0、V33a step 502 与 V34a step 1002；若墙体、远树和雪面覆盖确认不退化，再做一个新的受控平衡实验，使 opacity 删除预算受“事件前人口最低保留 95%”约束，而不是永久固定为当前人口的 5%，并继续监控最差 ROI 和纹理—密度相关性。

## 8. V35 屏幕足迹锐化短臂方案（2026-08-29）

用户对 V34a step 1002 PLY 的同视角检查指出：LiDAR 缺测附近仍存在模糊的大高斯，高斯外观偏大、偏圆，墙面和雪面细节不够锐利。数值复核支持该观察：可见高斯中投影足迹大于 `5 px` 的比例为 `39.4%`；最长轴大于 `2 cm` 的可见高斯有 `79,483` 个，其中轴比小于 3 的近圆大高斯为 `9,524` 个。与此同时，大量大高斯已经是高轴比薄盘，因此不能把“大于 2 cm”直接等价为错误点，也不能全局缩小或删除。

下一轮不修改竞品确认的 `0.2 m` world split 阈值。新增的 CloudStudio 细节分支只把同时满足以下条件的原 clone 候选改为 split：projected-gradient 高于既有阈值、opacity 高于 `0.15`、LiDAR 父点支持通过、最长轴大于 `2 cm`，且最近 100 step 的真实最大 raster radius 大于图像长边的 `0.35%`（全分辨率 1456 px 约为 `5.1 px`）。平滑、低梯度的大薄盘仍保留；小而清晰的点继续 clone。opacity 删除达到每轮 5% 上限时，排序也改为优先处理“低 opacity 且真实屏幕足迹大”的点，而非仅按世界尺度判断。

实验只做 Tile_1：A0 固定拓扑和 V34a step 1002 直接复用，不重复消耗 GPU；V35a 从相同初始化独立跑到 step 1002，只新增上述屏幕足迹条件 split/cull priority，其他 loss、PPISP、LiDAR 守卫和 observation-aware cull 均保持不变。只有 V35a 同时满足墙体/远树/雪面 alpha 不低于 V34a、LiDAR 距离 P95 不超过 `1 cm`、无 `>0.3 m` 漂移点、投影足迹 `>5 px` 比例明显下降、固定 ROI 清晰度和 SSIM 不退化，才允许继续 2618。竞品的 opacity-mean `0.01` 不在 V35a 同时开启；我方缺少竞品 DA2/mesh 的同等有效监督，直接把当前 `1e-4` 放大 100 倍会把锐化与容量稀疏化混为一个变量，并可能重新造成墙体透明。若 V35a 锐化通过但低贡献雾点仍高，再用独立 V35b 单变量短臂定标 opacity-mean。

当前状态为 `V35_SCREEN_DETAIL_ARM_IMPLEMENTED_NOT_RUN`。代码门禁和相关回归已通过，尚未启动训练、未生成新的 PLY，也未把该增强宣称为竞品原样行为。

## 9. V35–V38 复盘与竞品语义重新收敛（2026-08-29）

问题现象：V35 的普通 projected-gradient 只得到弱纹理趋势；V36a 删除过强，V36b 虽把 Tile_1 短程 PSNR 提到 `7.047 dB`，但纹理—密度相关仅 `rho=0.0037`，opacity 中位数只有 `0.0368`，约 `72.7%` 高斯低于 `0.1`，真实渲染 alpha 仍低。V37a 的比例式压平将可见短轴中位数从约 `3.49 mm` 降到 `2.75 mm`、轴比从约 `2.14` 提到 `2.82`，证明形状损失有效，但没有解决透明和容量重分配。随后准备的 V38a 叠加 LiDAR alpha 目标与 2 cm 局部累计 alpha 保护，仍是在自定义世界体素中代理真实渲染覆盖，不能视为竞品机制。

复盘确认的根因：当前 Trainer 顺序为 `backward -> Gaussian Adam step -> Split/Clone/Cull`，竞品静态审计恢复的是 `backward -> Split/Clone/Cull -> Gaussian Adam step`；当前 V36/V37 还启用了竞品没有证据支持的 LiDAR 硬出生守卫、局部 Cull 保护、单轮删除上限、detail split、薄盘目标，并使用 `revised_opacity=true`，而竞品 Split 子点直接重复父 opacity。现有 PPISP 也不是竞品 `[N_camera,12,8,16,16]` BilateralGrid；DA2、mesh depth/normal 与 SIFT 仍未形成等价消费闭环。因此同名 `1.5e-4` 梯度阈值不能被宣称为等价定标。

修改文件：`cloudstudio_3dgs/training/default_strategy_adapter.py`、`cloudstudio_3dgs/training/lidar_normals.py`、`cloudstudio_3dgs/training/trainer.py`、`cloudstudio_3dgs/pipeline/mipmap_gate.py`、`tools/build_v26a_boundary_gate.py`、`tools/promote_v26a_evaluation.py` 及对应测试。上述代码保留为可关闭的研究能力；LiDAR alpha 和局部累计 alpha 不进入下一条竞品语义主臂。

验证与停止证据：定向回归为 `143 passed, 1 deselected`；排除项只依赖当前未加载的 3DGUT 栅格器，不是本轮回归失败。V38a 在约 step 240 被强制终止，尚未到 step 500 首次拓扑事件，只有监控日志，不构成有效实验、checkpoint 或晋级证据。

优化决策：下一版先实现并单测签名的 `pre_optimizer` lifecycle；必须显式处理拓扑变换后的参数梯度与 Adam state，不能简单移动函数调用导致当前步梯度丢失。同时按竞品顺序执行 `Split -> Clone -> Cull -> Reset`，关闭 `revised_opacity` 和所有 CloudStudio Cull/alpha/薄盘补丁，形成厂商语义臂。BilateralGrid、dense mesh/normal、DA2 分别作为后续正交 A/B，不与生命周期时序修复同轮加入；其中 DA2 仍保持可选，不能因竞品使用就覆盖 LiDAR 权威几何。

当前状态：`V38A_ABORTED_VENDOR_SEMANTICS_REWORK_REQUIRED`。禁止继续 V36/V37/V38 长训，下一步先完成 CPU 梯度/optimizer-state 变换测试和 2-step GPU smoke，再跑 Tile_1 step 502；通过后只延长到能经历 reset 后下一次 Cull 的短 review，不直接进入全 Tile。

## 10. V40–V46 生命周期定标结论（2026-08-29）

用户对 V40a step 502 PLY 的视觉反馈是雪堆略透明、细节提升不明显。全 374 个 Tile_1 Face4 视图复核证明，问题不是“训练步数不够”：V40a step 502 为 PSNR `10.34543 dB`、alpha `0.57453`；V42b step 602 通过普通梯度阈值 `7.5e-5` 提升到 `10.42081 / 0.59933`，但同路线继续到 step 1002 后又明显下降。

梯度和 Cull 遥测给出了可重复的原因。step 600 普通 projected-gradient 的 P50/P95/P99 为 `2.11e-5 / 7.71e-5 / 1.50e-4`，因此竞品阈值 `1.5e-4` 在我方 PPISP/LiDAR-only 环境只选择约最高 1% 父点；`7.5e-5` 才接近高纹理 5% 分位。另一方面，即使把 opacity sparsity 从 `0.01` 降为 `0.001`，使用 `0.04` Cull 的 V44a 到 step 1002 仍有 PSNR `9.54298`、alpha `0.58510`；完全关闭 opacity Cull、只保留 world/screen 异常删除的 V45a 也仅为 `9.64908 / 0.59199`。这说明 600 步后的损失来自持续 opacity/光度/拓扑联合优化，不只是 Cull 删除数量。

当前通过的组合是 V45b step 602：从相同 V40a step 502 checkpoint 只再训练 100 步，使用普通梯度 `7.5e-5`、deferred reset、可见高斯 opacity sparsity `0.001`，关闭 opacity 阈值删除但保留 world/screen 尺度异常删除。step 600 clone `39,170`、split 父点 `87`、几何异常 cull `767`，事件后为 `863,951` 个高斯。374 视图指标为 PSNR `10.45167 dB`、alpha `0.60189`、alpha P05 `0.08731`、alpha 不足比例 `0.60039`，是当前同口径最优节点。

screen-aware Split 没有通过。V46a 把同一批候选中的 split 父点从 `87` 提高到 `5,267`，总点数保持相同，但 PSNR/alpha 下降到 `10.27435 / 0.58494`；启用 revised opacity 的 V46b 进一步下降到 `10.17067 / 0.57562`。因此“大屏幕高斯直接 Split”会在当前阶段损坏覆盖，不能用作锐化捷径。

V45b 完整 PLY 位于 `outputs/snow-20260224-full-20260825/v45_geometry_cull_only/training_tile1_v45b_growth7p5e5_geometrycull_opacity1e3_review602/exports/snow_tile1_v45b_step602_sh0_full.ply`，包含 `863,951` 个 SH0 Gaussian，大小 `58.7 MB`。最长轴 P50/P95 为 `7.4/18.0 mm`，最短轴 P50 为 `3.5 mm`，轴比 P50 为 `2.14`；可见高斯中仍有 `25.3%` 宽于 5 px。因此它是覆盖和短程视觉指标的当前最佳，不是最终竞品级薄盘终态。

当前决策为 `V45B_STEP602_CURRENT_TILE1_CHAMPION_LONG_RUN_BLOCKED`：保留并导出 V45b，V44/V45a/V46 均只作为失败或诊断证据，不启动长跑和全 Tile。下一项研究只能单变量处理形状/光度，不得把 MCMC、DA2、天空、opacity Cull 和 screen-detail Split 同时重新加入。

## 11. V47–V48 单变量覆盖与形状验证（2026-08-29）

问题现象：V45b 在白背景上具有当前最佳短程指标，但白雪可通过降低 opacity 或缩小高斯足迹露出白背景来减小 RGB 误差，SuperSplat 的非白背景因此会暴露透明洞。原 Face4 验证又没有加载真实 LiDAR sidecar，不能量化 LiDAR 命中区域的 alpha。另一方面，现有高斯最短轴中位数约 `3.57 mm`、轴比约 `2.16`，仍显著厚于竞品薄盘，模糊和覆盖问题必须解耦处理。

修改文件：`cloudstudio_3dgs/pipeline/mipmap_gate.py`、`cloudstudio_3dgs/training/checkpoint.py`、`cloudstudio_3dgs/training/ppisp.py`、`cloudstudio_3dgs/training/trainer.py`、`tools/build_v47_surface_alpha_probe.py`、`tools/build_v48_surface_shape_probe.py`、`tools/evaluate_checkpoint_validation.py`、`tools/promote_v26a_evaluation.py`、`tests/test_mipmap_gate.py`、`tests/test_training.py`。

修改内容：Face4 验证正式消费配置绑定的稀疏 LiDAR range；短实验统一使用同一组确定性分层 48 视图，黑背景仅抽 8 视图，全 374 视图只留给晋级候选。V47 只训练 opacity 并冻结其余参数，用 dilate-3 的 LiDAR alpha 目标排除白背景捷径。V48 以 V47e step 652 为不可变起点，关闭后续 lifecycle，冻结点数、means、SH0、opacity 与 PPISP，只训练 scales/quats；比例压平仅作用于可信平面 LiDAR anchor，LiDAR alpha 继续维持真实表面覆盖。

结果：V47e 的 48 视图 PSNR/alpha/LiDAR alpha 为 `10.3004/0.62386/0.95366`。V48a 的 `0.01` 压平权重只使轴比中位数到 `2.177`，信号太弱。V48b 将 scales/quats LR 定标为 `0.003/0.0005`，法线对齐和压平权重为 `0.1/0.1`，50 步后点数仍为 `863,951`，means、SH0、opacity、PPISP 均逐值不变；中位最短轴降到 `2.963 mm`、轴比升到 `2.599`。固定 48 视图 PSNR/alpha/LiDAR alpha 为 `10.5218/0.63623/0.95679`，黑背景 8 视图也从 V47e 的 `10.4260/0.57452/0.95542` 提升到 `10.5031/0.58502/0.95905`。说明提升来自高斯形状和投影覆盖，不是删点、降 opacity 或白背景作弊。

当前判断：`V48B_NUMERIC_PASS_VISUAL_PENDING`。它已经优于 V47e 的覆盖和锐化代理，但中位轴比仍远低于竞品终态，且深度 MAE 随投影足迹变化轻微上升到 `0.5639 m`。在 SuperSplat 固定视角确认墙体、雪堆和远景没有新增孔洞前，不继续增加权重、不恢复 projected-gradient lifecycle，也不扩展全场。候选 PLY：`outputs/snow-20260224-full-20260825/v48_surface_shape/training_tile1_v48b_scalequat_strong_ratio0p15_review702/exports/snow_tile1_v48b_step702_shapeonly_sh0_full.ply`。

## 12. V62 连续 LiDAR mesh 与 DA2 独立消费 A/B（2026-08-29）

问题现象：V61 全场结果仍有透明墙面、地面空洞和接缝；既有 Trainer 只有约 3%–5% 像素的稀疏 LiDAR range，不能在 LiDAR 射线之间提供连续表面和法线监督。竞品审计确认其把逐视图 mesh depth、mesh normal 与经 mesh 鲁棒米制定标的 DA2 分成三路 loss，并按完整视图 epoch 分阶段启停，而不是融合成一张硬深度。

修改文件：`cloudstudio_3dgs/data/mesh_geometry.py`、`cloudstudio_3dgs/training/dataset.py`、`cloudstudio_3dgs/training/face_dataset.py`、`cloudstudio_3dgs/training/trainer.py`、`cloudstudio_3dgs/training/mipmap_loss_schedule.py`、`tools/build_lidar_mesh_candidate.py`、`tools/rasterize_lidar_mesh_face4_probe.py`、`tools/build_da2_mesh_aligned_tile_manifest.py`、`tools/audit_mesh_cross_view_consistency.py`、`tools/build_v62_dense_geometry_smoke.py`、`tools/build_v62_dense_geometry_ab.py` 及对应测试。

修改内容：Trainer 现可校验并独立加载签名的 `mesh_depth + mesh_normal + confidence + valid` sidecar；mesh depth 使用竞品恢复的远距离压缩函数，mesh normal 使用渲染深度重建法线的兼容代理并做半球对齐 L1。DA2、mesh depth、mesh normal 的 loss、有效权重和 mask 覆盖均进入运行遥测。竞品 `[5V,10V,5V]` schedule 保留 20 个完整 view epoch 门禁，同时允许配置权重对每一路进行缩放或关闭，使未经验证的 DA2 不会阻断已验证的 LiDAR mesh 主线。当前 Open3D BPA 只标记为候选连续表面算法，不能宣称是竞品未恢复的建网拓扑。

Tile_1 使用 374 张全分辨率 Face4，固定 `971,903` 个高斯、相同 seed 和视图顺序完成 100-step A/B。对照臂 24 视图为 PSNR `11.2206 dB`、SSIM `0.559247`、LiDAR depth MAE/RMSE `4.139/12.038 cm`、alpha `0.641504`。三路从 step 1 满权重开启时，depth MAE 改善到 `3.403 cm`（`17.8%`），但 PSNR/SSIM 降到 `10.9886/0.554837`，alpha 降到 `0.625861`。关闭 DA2、只保留 mesh depth `0.5` 与 mesh normal `0.05` 后，depth MAE/RMSE 为 `3.675/11.147 cm`，相对对照改善 `11.2%/7.4%`；PSNR 仅变化 `-0.011 dB`、SSIM `-0.000313`、alpha `-0.002065`。加权 loss 审计证明退化主要由当前 DA2 `0.5` 主导，而不是 mesh depth。

验证方式：mesh/DA2/调度/Trainer 定向回归 `71 passed`；四个 100-step 训练均完成且点数严格不变，峰值显存约 `2.06–2.40 GiB`；固定 24 视图按签名顺序完成 metrics-only 对照。当前状态为 `V62_MESH_ONLY_NUMERIC_PASS_DA2_BLOCKED`。mesh-only 可以接入后续固定拓扑分阶段主线；DA2 必须先补生产级跨视图一致性置信度、深度边界/遮挡门禁和失败视图剔除，不能直接以 `0.5` 晋级。正式长训和 adaptive Grow/Cull 仍未授权通过本轮短 A/B。
