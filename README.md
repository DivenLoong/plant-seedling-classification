# 任务一：植物分类 —— 训练与提交流水线

Kaggle 竞赛 `2026task1`，评分指标为 F1（实际等价于 378 张测试图上的准确率，
由排行榜分数差恒为 1/378 的整数倍可以反推出这一点）。

## 数据

| 项目 | 内容 |
| --- | --- |
| 训练集 | 5 类各 100 张，共 500 张，文件名已匿名化 |
| 测试集 | 378 张，文件名已匿名化 |
| 类别 | Black-grass / Common wheat / Loose Silky-bent / Scentless Mayweed / Sugar beet |
| 分辨率 | 49×49 到 2840×2132 不等（原图混合分辨率，来自 Aarhus 大学公开植物幼苗库） |
| 提交格式 | `ID,Category`，378 行 |

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `plantcls.py` | 公共库：数据扫描、前景裁剪、数据集、骨干网络、训练与推理引擎 |
| `eda.py` | 数据集统计、前景裁剪覆盖率、归一化参数、样例对照图 |
| `build_cache.py` | 预解码全部图像并缓存裁剪结果（消除训练时的 IO 瓶颈） |
| `train.py` | 分层 K 折训练，输出每折权重、训练历史、OOF 概率与 summary |
| `predict.py` | 多模型 + 多视图 TTA 集成推理，生成 `submission.csv` |
| `blend.py` | 组合多个 run，估计测试集类别先验并验证其有效性 |
| `report.py` | 训练曲线、混淆矩阵、逐类指标，汇总到 `outputs/report.md` |
| `errors.py` | 把 OOF 误分类样本拼成对照图，用于分析错误来源 |
| `dupes.py` | 检查训练/测试之间是否存在近似重复（排除数据泄漏） |
| `recon.py` | 最初的数据集探察脚本 |

所有产物写入 `outputs/`：`<tag>/` 存放每个实验，`imgcache_*` 存放图像缓存，
`report.md` 与 `blend.json` 汇总结论。

## 方法

1. **前景裁剪与尺度归一化**：用 HSV 绿色掩膜求出植物包围盒，裁剪后补成正方形。
   原图分辨率差异极大，这一步把尺度统一，同时去掉大部分土壤背景。
   9 张带 alpha 通道的 PNG 会被透明边框污染，专门做了合成处理。
2. **预训练骨干微调**：ImageNet / ImageNet-21k 预训练模型，先冻结骨干只训分类头，
   再解冻全网络用 OneCycle 余弦退火微调。
3. **增强**：随机裁剪缩放、水平/垂直翻转、任意角度旋转（俯拍图旋转不变）、
   RandAugment、Random Erasing、Mixup + CutMix、label smoothing。
4. **5 折分层交叉验证**：每折训练一个模型，测试时把 5 折的预测概率平均；
   OOF 概率只用于评估，训练时保留最终权重而非"按验证集挑最优轮次"，避免选择偏差。
5. **多尺度多旋转 TTA**：2 个尺度 × 4 个 90° 旋转共 8 个视图，概率平均。
6. **半监督伪标签**：用集成模型给测试图打高置信度（≥0.9）伪标签，加入每折训练集后重训。
7. **类别先验校正**：训练集每类各 100 张是均衡的，测试集却按原始公开库的类别比例分布。
   在准确率指标下，最优决策是 `argmax_c π_c · p(c|x)`。先验由 OOF 混淆矩阵反演估计，
   并在重采样的 OOF 上验证过该估计流程的准确性。

## 复现步骤

```bash
python eda.py                 # 数据统计（生成归一化参数）
python build_cache.py         # 预解码图像缓存
python train.py --tag cnx320 --arch convnext_tiny.fb_in22k_ft_in1k --size 320 \
    --head-epochs 4 --ft-epochs 40 --raug --mixup 0.2 --cutmix 1.0 \
    --weight-decay 0.05 --label-smoothing 0.1 --lr 2e-4
python predict.py --tags cnx320 --out submission_v1.csv      # 生成 test_probs.npz
python train.py --tag cnx320p --arch convnext_tiny.fb_in22k_ft_in1k --size 320 \
    --pseudo outputs/test_probs.npz --pseudo-thresh 0.9 <其余同上>
python blend.py --tags b0_288,cnx320,cnx320p
python predict.py --tags b0_288,cnx320,cnx320p --prior-json outputs/blend.json \
    --out submission.csv
python report.py --tags b0_288,cnx320,cnx320p
```

## 结果

5 折交叉验证的 out-of-fold 准确率（500 张训练图，每张都由没见过它的模型预测）：

| 实验 | 配置 | OOF 准确率 |
| --- | --- | --- |
| b0_288 | EfficientNet-B0 @288，普通增强 | 0.9140 |
| cnx320 | ConvNeXt-Tiny(in22k) @320，RandAugment + Mixup/CutMix | 0.9180 |
| cnx320p | 同上 + 伪标签（+192 张高置信度测试图） | 0.9280 |
| cnx320m | 同 cnx320p + 去背景（HSV 掩膜涂色） | 0.9140 |
| cnx384m | 同 cnx320m + 输入 384 | 0.9240 |
| cnxs320m | ConvNeXt-Small(in22k) @320 + 去背景 + 伪标签 | 0.9240 |
| rn50b | ResNet50(a1_in1k) @320 + 去背景 + 伪标签 | 0.9000 |
| b0m | EfficientNet-B0(ra_in1k) @320 + 去背景 + 伪标签 | 0.8840 |
| **6 组集成（v3 提交）** | 30 个模型 + 16 视图 TTA + 类别先验校正 | **0.9300** |
| 5 组集成（v2 提交） | 25 个模型 + 16 视图 TTA + 类别先验校正 | 0.9380 |

集成收益随模型数量单调上升：0.9140 → 0.9220 → 0.9280 → 0.9340 → 0.9380。

**关于集成成员选择的重要说明**：上述 0.9380 与 0.9300 的差别只对应 4 张图，
而 500 张样本上准确率的标准误约 1.1%，两者在统计上不可区分；按小数点挑选子集
等于拟合噪声。因此最终提交采用一条预先定好的规则——**纳入所有单组 OOF ≥ 0.91 的
模型**（b0m 与 rn50b 因此被排除）。另外测试集上每张图由 30 个模型投票，
而 OOF 中每张图只有 5 个模型投票，所以 OOF 会低估集成的真实收益。

逐类 F1：Scentless Mayweed 1.000、Sugar beet 0.995、Common wheat 0.970、
Loose Silky-bent 0.871、Black-grass 0.854。
可见**剩余错误几乎全部集中在 Black-grass 与 Loose Silky-bent 的互混**上。

线上成绩：v1 提交（15 模型 + 8 视图 TTA + 先验校正）为 **0.9417**。

### 已验证无效的方向

* **FixRes（换推理分辨率）**：320 训练、352/384/416 推理分别为 0.9280/0.9260/0.9260/0.9180，
  训练分辨率已是最优，这条排除。
* **EfficientNet-B3（NoisyStudent 权重）@320**：第 1 折只有 0.86，明显不适应本配方，已终止。
* **近似重复泄漏**：测试图与训练图的近似重复率（0.98 阈值下 3.2%）与训练集内部自然重复率
  （4.8%）相当，未发现可利用的泄漏。

## 同类数据集的公开工作（文献调研）

本数据集是 Aarhus 大学公开植物幼苗库（Giselsson 等，arXiv:1711.05440），12 个物种、
约 4275 张图、约 960 株植物、物理分辨率约 10 px/mm，论文提出的官方基准就是 **F1**。
本赛题把它裁剪成 5 类。

| 来源 | 方法 | 报告结果 |
| --- | --- | --- |
| JIFS 2021, *A novel two-stage method of plant seedlings classification* | 改进 U-Net 分割 + 六个分类网络 | 准确率 97.7% |
| ETASR 2022, *Deep CNN Architecture for Plant Seedling Classification* | EfficientNet-B4 / B2 迁移学习 | B4 准确率与 F1 99.0%，B2 97.0% |
| JPCS 2022, *Classification of plant seedlings using deep CNN architectures* | ResNet50V2 / MobileNetV2 / EfficientNetB0 | EfficientNetB0 准确率 96.5%、F1 96.3% |
| IJAEIS 2020, *Plant-Seedling Classification Using Transfer Learning* | ResNet50/VGG16/VGG19/Xception/MobileNetV2 | ResNet50 最优，95.2% |
| WuZhuoran 等（Georgetown 课程项目） | 自建 CNN + 去背景 + 过采样 + 快照集成 | 9 层 CNN 0.987；Kaggle 当时最好 0.995 |
| deepblacksky（GitHub） | HSV 绿色掩膜 + Xception/DenseNet121 两阶段微调 + 集成 | 去背景后精度提升、收敛更快；集成再涨 1–5 个点 |

由此得到三条可直接借鉴的结论：

1. **先去背景再分类**是最稳定的增益来源（HSV 掩膜是 U-Net 分割的廉价替代）。
2. **更大的骨干 + 更高输入分辨率**收益明显（B4 优于 B0 约 2.5 个百分点）。
3. **集成**是最后 1–3 个百分点的来源。

注意这些报告多数是在 **4275 张全量数据**上得到的；本赛题只提供 500 张训练图，
因此绝对精度不能直接对标，但方法取向依然成立。
