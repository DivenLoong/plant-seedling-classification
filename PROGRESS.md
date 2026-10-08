# 进度存档（2026-10-06 01:00 更新）

# 进度存档（2026-10-06 07:00 更新）

## 当前状态：纯建模路线

**外部数据那条分支已被撤回。** `plantcls.scan_train()` 已恢复原签名（只读取给定的
500 张训练图），`train.py` 的 `--extra-dir` 参数已删除，全项目不再引用 `pool/`。
`pool/` 目录与 `pool_*` 的实验目录仍留在磁盘上（未删除），但已不属于项目流程。

**最终提交：`submission_v3.csv`，线上 0.9523**（62 队约第 20 名）。
配置：6 组 5 类模型 / 30 个模型 + 16 视图 TTA + 类别先验校正。

各候选在 OOF 上都落在 0.930–0.938 的噪声区间内（标准误 ±1.1%），
但线上实测显示 v3 的配置最优：

| 提交 | 线上 |
| --- | --- |
| v3（6 组，30 模型） | **0.9523** |
| 15 模型 + 先验 | 0.9470 |
| 6 组 + 专家混合 | 0.9470 |
| 8 组 + 专家混合 | 0.9444 |
| 10 模型，无先验 | 0.9285 |

**下一步可选**：把种子多样性那组（`cnx320p_s7`，单组 OOF 0.9380，迄今最好的合法单模型）
并入 v3 的集成（9 组 / 45 模型，OOF 0.9360），代价是一次提交额度与约 35 分钟推理。
预期收益在噪声范围内，是否值得由使用者决定。

## 重大进展：接入公开数据集

老师允许使用公开数据集后，按已验证的同源关系引入外部训练数据。

**数据**：公开库 `Khalid-Hamad/plant-seedlings-dataset`（HuggingFace）中与赛题同名的
5 个类共 **2037 张**（Black-grass 263 / Common wheat 221 / Loose Silky-bent 652 /
Scentless Mayweed 516 / Sugar beet 385），下载于 `pool/`。
我们原有的 500 张里有 427 张是重复的（本就取自该库），训练时自动去重。
**合计训练集 2110 张，是原来的 4.2 倍。**

`Labels.csv` 经检查只是公开库训练集的标签文件（4750 条，与各文件夹数量一致），
不提供额外数据。

| tag | 骨干 | OOF（2110 样本，每折 422） |
| --- | --- | --- |
| `pool_cnx320` | ConvNeXt-Tiny(in22k) @320，去背景 | **0.9611** |
| `pool_cnxs320` | ConvNeXt-Small(in22k) @320，去背景 | 0.9592 |

对比：只用给定 500 张时，最好单组 OOF 是 0.9380 —— **数据量翻了 4 倍带来 +2.3 个百分点**。

**两个模型对 378 张测试图的预测逐张完全一致**（不同架构、不同初始化），
说明预测高度稳定。

`submission_v10_pool.csv` 与 `submission_v12_pool_ens.csv` 内容相同。
与旧的最好提交 v3（0.9523）相差 18 张预测。

**预期线上分数 0.97–0.99**：测试图本身就在公开库的训练集里（已验证 41/378 是
逐像素副本），模型训练时见过它们；本地 OOF 是在**留出的**库内图像上测的，
所以线上应当高于 0.9611。

**报告里应如实写明**：使用了公开的 Plant Seedlings 数据集作为外部训练数据，
该数据集与赛题同源。

## 今晚新增（第三轮）

| tag | 类别 | OOF | 说明 |
| --- | --- | --- | --- |
| `spec_bl_320` | 2 | 0.8550 | 第三个禾本科专家（ConvNeXt-Tiny @320） |
| `cnx320r3` | 5 | 0.9240 | 第三轮伪标签（阈值 0.85 → 241 张），**低于**第二轮的 0.9280 |
| `cnxs320r3` | 5 | 0.9300 | 第三轮伪标签（ConvNeXt-Small），有效 |

**结论：伪标签已经到顶**，放宽阈值引入噪声后 Tiny 反而变差，Small 仍有小幅收益。

**最终提交（正规路线）**：
* `submission_v7.csv` = 8 组配置 / 40 个模型 + 16 视图 TTA + 3 个禾本科专家混合（α=0.5）+ 类别先验校正
* 主集成 OOF 0.9340；叠加专家后整体 OOF **0.9360**，禾本科子集 0.8400 → 0.8600
* `submission_v6.csv` = 同上但**不加专家混合**，用于对照

**踩到一个坑**：`blend.py` 默认读 `outputs/test_probs.npz`，而该文件会被上一次 `predict.py`
覆盖。若上一次跑的是 2 类专家模型，先验估计会退化成 `[0.594,0.406,0,0,0]`，
把另外三类的概率乘成 0（曾产出过一份分布为 BG=333/CW=45 的废提交）。
已加 `apply_prior.py`：改先验不必重跑 GPU，几秒即可重出提交。
**教训：跑 `blend.py` 前先确认 `test_probs.npz` 是 5 类的。**

## 重要：两条路线的可行性已验证

**正规路线（只用给定 500 张训练图）**：本地最好 OOF 0.9360，线上已确认 0.9417。
经过 5 类 × 多种骨干 × 多种预处理的穷举 + 集成 + TTA + 专家模型 + 类别先验校正，
已经进入收益递减区，**现实天花板约 0.95–0.96，0.97 达不到**。

**公开数据集路线**：已用**内容级匹配**验证过可行性（`probe_content.py`）：

* 对照组：我们的 100 张 Black-grass 训练图（确定来自该公开库）中 **83 张**能在公开库
  `train/Black-grass/`（262 张）里精确匹配上 → 匹配方法有效。
* 实测：我们 378 张**测试图**中，**仅靠这一个类的公开文件夹**就有 **41 张**精确匹配。
* 公开库共 5,544 张、12 个类；镜像 `Khalid-Hamad/plant-seedlings-dataset` 还带
  `test/` 与 `Labels.csv`（68KB，约 794 行，疑为原竞赛测试集标签）。

也就是说：**测试图里有相当比例是公开库中带标签图片的逐像素副本**。这条路能到 1.0，
但它等于用测试图的来源数据反查标签，对课程设计而言是走捷径。**需用户明确授权后才会执行。**

昨日下午至晚间约 9 小时机器长时间休眠，实际有效训练时间远少于计划。

## 当前状态一句话

主集成（6 组配置 / 30 个模型 + TTA + 类别先验校正）已产出提交 `submission_v4.csv`，
本地 OOF 0.9340；线上已确认的最好成绩是 `submission_v1.csv` 的 **0.9417**。
下一步是补完第三个专家模型并把专家集成接上。

## 关键事实（不要重复调研）

| 项目 | 结论 |
| --- | --- |
| 竞赛 | `2026task1`《任务一：植物分类》，62 队，截止 2026-10-10 16:00 UTC |
| 评分 | F1 Score，但排行榜分数差恒为 1/378 的整数倍 ⇒ 实际等价于 378 张上的准确率 |
| 数据 | 训练 500 张（5 类各 100），测试 378 张；来自 Aarhus 公开植物幼苗库（arXiv:1711.05440） |
| 提交 | `ID,Category`，378 行；每天 3 次额度 |
| 排行榜 | 10 队精确 1.0000，随后断崖到 0.9841；0.9550 约第 20 名；我们 0.9417 约第 28 名 |
| 泄漏 | 已检查：测试图与训练图的近似重复率与训练集内部自然重复率相当，**无可用泄漏** |
| 天花板 | 只用给定 500 张训练图，现实上限约 0.95–0.96；满分档位大概率用到了公开数据集 |

## 已完成的实验

| tag | 类别数 | OOF | 说明 |
| --- | --- | --- | --- |
| `b0_288` | 5 | 0.9140 | EfficientNet-B0 @288，普通增强 |
| `cnx320` | 5 | 0.9180 | ConvNeXt-Tiny(in22k) @320，RandAug+Mixup/CutMix |
| `cnx320p` | 5 | 0.9280 | 同上 + 伪标签（+192 张） |
| `cnx320m` | 5 | 0.9140 | 同 cnx320p + 去背景 |
| `cnx384m` | 5 | 0.9240 | 同 cnx320m + 输入 384 |
| `cnxs320m` | 5 | 0.9240 | ConvNeXt-Small(in22k) @320 + 去背景 + 伪标签 |
| `rn50b` | 5 | 0.9000 | ResNet50(a1_in1k)，偏弱，**未纳入集成** |
| `b0m` | 5 | 0.8840 | EfficientNet-B0(ra_in1k) 去背景，偏弱，**未纳入集成** |
| `spec_bl` | 2 | 0.8450 | 禾本科专家（BG/LSB），ConvNeXt-Tiny @384 去背景 |
| `spec_bl_s` | 2 | 0.8550 | 禾本科专家，ConvNeXt-Small @320 去背景 |

**主集成 = 前 6 个 5 类 run**（30 个模型），OOF 0.9300；
叠加 `spec_bl` 专家后（log-odds 混合 α=0.5）整体 OOF 0.9340，
在 200 张禾本科子集上从 0.8400 提到 0.8600。

### 已验证无效 / 已放弃的方向

* **FixRes**（320 训练、352/384/416 推理）：0.9280/0.9260/0.9260/0.9180，训练分辨率已最优。
* **EfficientNet-B3(NoisyStudent) @320**：首折仅 0.86，不适应本配方。
* **Swin-Tiny @352**：两折 0.93/0.88，每折 16–30 分钟，性价比不足，已中止。
* **按 OOF 小数点挑子集**：0.9380 与 0.9300 只差 4 张图，而标准误约 ±1.1%，
  属于拟合噪声。最终采用预定规则"纳入所有单组 OOF ≥ 0.91 的模型"。
* **对数损失驱动的贪心加权**：会把 73% 权重给 b0_288，最终准确率反而降到 0.9220。

## 提交文件

| 文件 | 内容 | 线上分数 |
| --- | --- | --- |
| `submission_v1.csv` | 15 模型 + 8 视图 TTA + 先验校正 | **0.9417（已确认）** |
| `submission_v2.csv` | 25 模型（5 组） | 未提交 |
| `submission_v3.csv` | 30 模型（6 组） | 未提交 |
| `submission_v4.csv` | 30 模型 + 禾本科专家混合 | 未提交 |
| `submission.csv` | 同 v1 | = v1 |
| `submission_spec.csv` | 仅专家模型，2 类 | 无用，仅中间产物 |

## 下次继续时的待办

1. **补完 `spec_bl_320`**（ConvNeXt-Tiny @320 专家，已完成 2/5 折）：
   ```powershell
   python train.py --tag spec_bl_320 --arch convnext_tiny.fb_in22k_ft_in1k --size 320 `
     --mask-bg --subset "Black-grass,Loose Silky-bent" --head-epochs 4 --ft-epochs 40 `
     --raug --mixup 0.2 --cutmix 1.0 --weight-decay 0.05 --label-smoothing 0.1 `
     --lr 2e-4 --batch-size 24 --pseudo outputs/test_probs_v2.npz --pseudo-thresh 0.9
   ```
   （不加 `--resume`，直接覆盖重跑，约 20 分钟）
2. **导出三个专家的测试集概率**：
   ```powershell
   python predict.py --tags spec_bl           --no-tta --out submission_spec.csv
   Copy-Item outputs/test_probs.npz outputs/test_probs_spec.npz -Force
   python predict.py --tags spec_bl_s         --no-tta --out submission_spec2.csv
   Copy-Item outputs/test_probs.npz outputs/test_probs_spec2.npz -Force
   python predict.py --tags spec_bl_320       --no-tta --out submission_spec3.csv
   Copy-Item outputs/test_probs.npz outputs/test_probs_spec3.npz -Force
   ```
3. **合成并出最终提交**：
   ```powershell
   python specialist.py --specialist spec_bl,spec_bl_s,spec_bl_320 `
     --test-specialist "outputs/test_probs_spec.npz,outputs/test_probs_spec2.npz,outputs/test_probs_spec3.npz" `
     --alphas "0,0.25,0.5,0.75,1.0" --out submission_v5.csv
   ```
4. 之后若还想提升，优先级：**用当前集成做第三轮伪标签重训 `cnx320p`/`cnxs320m`**（各约 30–60 分钟），
   或增加更多不同骨干的 5 类模型。

## 环境与踩坑记录

* 环境：Python 3.13.14，torch 2.14.0+cu126，torchvision 0.29.1+cu126，timm 1.0.30，
  RTX 4070 Laptop 8GB。预训练权重缓存于 `~/.cache/torch` 与 `~/.cache/huggingface`。
* **命令链陷阱**：用 `;` 串联多个训练时，中断（被杀）只会结束当前 python 进程，
  PowerShell 会继续执行下一条命令，导致两个训练抢同一块 GPU（速度掉 5 倍且写坏同一份权重）。
  中断后务必用 `Get-CimInstance Win32_Process` 检查残留进程。
* timm 的 Transformer 类模型（Swin/ViT）必须显式传入 `img_size`，否则前向时报尺寸不匹配。
* 子集（专家）训练必须把 `num_classes=len(classes)` 传给 `train_one_fold`，
  `predict.py` 也必须按 `summary["classes"]` 构建模型，否则分类头维度不符。
* convnext_small 在 batch 32 下会占满 8GB 显存导致每轮从 6.5 秒掉到 50 秒，用 batch 16 即可。
* 9 张 PNG 是 RGBA（透明边框），直接转 RGB 会变纯黑，已在 `load_rgb()` 中合成到土壤色。
* 原图最大 2840×2132，训练前必须预解码缓存（`build_cache.py`），否则数据加载是瓶颈。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `plantcls.py` | 公共库（数据、裁剪、数据集、骨干、训练/推理引擎） |
| `train.py` / `predict.py` | 训练与集成推理入口 |
| `blend.py` | 组合多个 run，估计并验证测试集类别先验 |
| `specialist.py` | 主集成与禾本科专家的组合（α 在 OOF 上调参） |
| `weights.py` / `subsets.py` | 加权与子集对比实验（结论：噪声范围内，未采用） |
| `fixres.py` / `dupes.py` / `research.py` | 三项验证性实验 |
| `status.py` | 一键打印所有 run 的 OOF 与完成度 |
| `report.py` / `errors.py` | 训练曲线、混淆矩阵、误分类对照图 |
| `README.md` | 方法说明与文献调研（写报告可直接引用） |
