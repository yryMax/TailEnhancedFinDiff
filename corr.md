# Cross-Factor Correlation 实验综合分析

对比四组 factor_evaluation 实验，评估在 DLPM 上引入协方差结构的不同方式。

## 配置与指标总表

| 配置 | ckpt | levy_alpha | 修改 | Kurt MAE | Std MAE | Skew MAE | Fro | Corr scatter |
|---|---|:-:|---|---:|---:|---:|---:|---|
| **dev** | regression | 1.9 | 无 | **3.95** | 0.0011 | 0.71 | 1.59 | 塌缩到 0 |
| **ddpm_with_L** | DLPM_corr_alpha20_L | 1.9 | 彩噪 `eps=sqrt(A)·Lz` | 555.87 | 0.0045 | 12.37 | 9.99 | 有趋势但不一致 |
| **white** | DDPM/white | 2.0 | 数据白化 `Y=X·L⁻ᵀ` | 17.42 | 0.0012 | 0.80 | **1.55** | **最贴对角线** |
| **white_levy** | DLPM_corr_alpha19_white | 1.9 | 白化 + Lévy | 1997 | 0.17 | 35.16 | 4852 | 整体崩 |

对照基线：Resample Fro≈0.95，Gaussian Fro≈0.89，train(vs OOS) Fro≈0.89。

---

## 现象解释

### 1. dev：marginal 好、corr 烂

α=1.9 DLPM 天然匹配金融因子重尾。但 denoiser 目标 `eps = sqrt(A)·z` 各维度独立，MSE 损失各向同性——**训练目标里没有写入 C 的信息**，因子间协方差只能靠数据耦合到输入 x 这条弱通道。结果：边际分布精确，联合分布塌缩到 near-diagonal。

### 2. ddpm_with_L：Kurt 拉爆

目标变成 `eps = sqrt(A)·Lz`，两种各向异性相乘：
- C 的小特征值方向 MSE 权重低 → 欠学习
- Lévy `sqrt(A)` 可达 ~45 → 放大欠学习方向的误差
- reverse 里 `1/gamma_t` 累积 T=100 步

少数样本在 C 的某些方向飞出去：Kurt=555、Fro=9.99。Corr scatter 反而更贴对角线，说明 L 确实改了协方差、信息是学进去了，**但被 tail outlier 彻底污染**。

### 3. white (α=2.0)：唯一可用的改进方案

白化把训练目标变成 Cov=I 的良态问题 → MSE 学得干净 → unwhiten `X_gen = Y_gen @ L^T` 复原协方差。**数学上最干净的路线**：
- Fro=1.55 与 dev 持平
- Corr scatter 第一次出现明显的对角线趋势
- 代价：unwhiten 在 C 的大特征值方向（market）放大残差，Kurt 从 3.95→17.4

这是四个方案里 corr/marginal 权衡最合理的。

### 4. white_levy (α=1.9 + 白化)：彻底崩溃

两个增强项乘性冲突：
- 白化后输入空间 Cov≈I，**但 Lévy A 仍带来极端值** → 训练不稳
- unwhiten 时 `L^T` 把白空间里的极端样本投射到原始空间主导方向（market Std 从真实 0.01 飙到 0.46）
- Kurt 到 3000、Fro 4852

---

## 其他发现

- **dev 的 train-Fro=0.89 vs gen-Fro=1.59**：即使"稳定版本"，DLPM 学到的协方差也比训练集本身更偏。说明 baseline 本身不捕捉 corr，不是测试集问题。
- **white 的 Mahalanobis Q-Q**：Generated 绿点几乎贴 x 轴——模型生成样本的 Mahalanobis 距离远小于真实，**联合分布偏"瘦"**。这和 corr scatter 改善并不矛盾：pairwise corr 对，但整体 joint 缺重尾密度。
- **white_levy 的 KDE**：几乎所有因子一根极窄尖峰 + 稀疏极端值——**典型的模式塌缩 + outlier 混合**。KS p-value 在 size/quality 高（0.89/0.25）不是因为好，是分布过度集中导致 KS 失效。
- **mar-vol pair**：三个非崩溃方案都抓到 market-volatility 的强负相关（-0.5 左右），因为是因子数据里信噪比最高的一对，所有方法都能学。真正区分方案的是 |ρ|<0.3 的中间对。
- **ddpm_with_L 的 corr scatter**：尽管 Fro 大，绿点分布比 dev 更贴对角线——说明 L 路线**不是错了，是不稳**。

---

## 核心结论：Lévy 与 Corr 增强项乘性冲突

两个增强项在当前设定下冲突：
- Lévy 要求对极端 A 鲁棒
- 白化 / L 彩噪要求忠实还原各向异性结构

叠加时会把"偶发极端样本"精准投射到"协方差主导方向"，所以 white_levy 比任一单项都差得多。

---

## 可行路线排序

1. **white α=2.0** — 唯一联合分布改善且未崩的方案。继续调优：降 lr / 加 kurtosis-robust loss / 限制 `L^T` 条件数。
2. **α=1.9 + L 彩噪若想救**：
   - whitened-L（等价方案 3 但保留 Lévy subordinator）
   - anisotropic loss（按 `1/λ_i` 加权 MSE，补偿小特征值方向）
3. **放弃同时捕捉 corr 和重尾**：两阶段方案——α=1.9 学 marginal，事后用 copula / rank 变换接经验 corr。

## 代码路径

- 彩噪：`factor_diffusion_train.dlpm_loss` 接 `L` 参数；`factor_diffusion_sample.generate` 初始化与噪声注入乘 `L.T`
- 白化：`factor_diffusion_train.__main__` 里 `X_white = X @ L_inv.T`；`generate()` 出口处 `out = out @ L_whiten.T`
- 互斥：`USE_L_NOISE` 与 `WHITEN` 在 `__main__` 里 raise ValueError
