# Data-Scaling Ablation（训练集规模 / 学习曲线）

实验代号：**`data_scaling`**

## 命题（Hypothesis）

> 增大训练数据量能否提升 DLPM 在 `factor_evaluation` 上的生成质量？

度量沿用 `factor_evaluation.ipynb` 那一套
只需要算出四个marginal statistics 和 covariance frob然后列表

---

## 现有素材

**「增长数据集」= 增长 factor.csv 的行数（更多交易日）**，不是更多股票。

**大 csv 已经存在** = `model/full/factors.csv`（2001–2025，6522 行，date 索引，列 = `alpha + 7 因子`，每年约 261 行）。子集化只是按 date 切行，**不必重跑 factor model**；若要刷新重生成一次：`python factor_model.py full`。
---

## ⚠️ 设计关键点：你说的「增量训练」要拆成两种，别混

| 方案 | 做法 | 验证的命题 | 用不用 |
|---|---|---|---|
| **A. 学习曲线（从头训子集）** | 对 25%/50%/75%/100% 数据各**从随机初始化训到收敛**，比指标 | 纯「数据量 → 性能」 | ✅ **推荐主方案** |

选择A
---

## 主设计：Expanding-Window Walk-Forward（扩张窗口滚动）

每折的 OOS 都不同：**train 截止年 Y → OOS = 紧随的 Y+1 年**。
- 第一折：train `2001–2005` → OOS `2006`
- 最后一折：train `2001–2024` → OOS `2025`（当前）
- 共 **20 折**，训练窗口从 5y 扩张到 24y，逐折增加最近一年。
> 注：源数据 2001–2025，没有 2026；草稿里的「2000/2026」按上表修正。

**指标（5 个，列表对比）**：`Mean MAE` · `Variance MAE` · `Skewness MAE` · `Kurtosis MAE` · `Cov Frobenius`。
前四个是 marginal 矩在 7 个因子上的 MAE，第五个用 `metrics.statistic.compute_cov`。

---

## 实验协议（Protocol）

**固定不变（控制变量）：**
- `num_generate=4096`；模型结构、`levy_alpha`、`num_timesteps`、`batch_size`、`lr` 全 = 当前 eval cfg。
- **训练步数对齐**：小窗口（5y）也要训到收敛，否则把「欠拟合」误读成「数据不足」。按**总 gradient steps 对齐或各自训到 loss 平台**，记录每折收敛曲线。

**变量：**
- 训练窗口年数 ∈ {5, 6, …, 24}（20 折），对应测试年 {2006, …, 2025}。
- seed ∈ {0,1,2}（小窗口方差大，多 seed 取均值±std；先 1 seed 快速看趋势也行）。

**输出：**
- 主表：`fold → {训练年数, 测试年, 指标们}` mean(std)。
- 主图：指标 vs 训练年数曲线 

---

## 实现（一次性脚本，无 baseline）

单文件 `ablations/data_scaling.py`：20 折扩张窗口，每折切 `model/full/factors.csv` → 复用
现有训练脚本从头训 → `DiffusionSampler(ckpt).generate(4096)` → **只和该折自己的 OOS 年比**
5 个矩指标 → 存 `data_scaling_results.csv`（表）+ `data_scaling.png`（指标 vs 训练年数，一图）。
不算 Resample/Gaussian baseline。throwaway，不做兼容性设计。

```
python ablations/data_scaling.py --fold 1   # 单折冒烟
python ablations/data_scaling.py            # 全 20 折
```

<details><summary>原始细化清单（保留）</summary>


**原则：增量修改，零改动现有 `.py` 接口。** 只新增 `ablations/` 下 3 个文件 + 一个 scratch
实验目录 `model/data_scaling/`。每折切片成 csv + 写一份 cfg，**直接复用** 现有
`factor_diffusion_train.py` / `DiffusionSampler` / `ResampleSampler` / `GaussianSampler` /
`metrics.statistic.compute_cov`。

### 复用点（已确认，不用改）
- `factor_diffusion_train.py <exp>`：读 `model/<exp>/cfg.yaml` 的 `data_file` → `load_data`
  在该 csv 上 `StandardScaler().fit`（→ scaler 自动只拟合在**该折训练窗**，正是我们要的）
  → 训练 → `torch.save({model_state, model_kwargs, scaler, cfg, ...})`。
- `DiffusionSampler(ckpt).generate(N)`：从 ckpt 自带 scaler/cfg，返回**原始单位** (N,7)。
- `ResampleSampler(train_df[FACTORS], scaler).generate(N)` / `GaussianSampler(...)`：同折基线。
- `compute_cov(gen, oos)`：相对 Frobenius 距离，一行。

### 文件 1：`ablations/eval_moments.py`（指标函数，~25 行）
```python
import numpy as np
from scipy.stats import skew, kurtosis
from metrics.statistic import compute_cov

def evaluate(gen, oos):          # gen:(N,7)  oos:(M,7)  原始 factor returns，列同序
    mae = lambda a, b: float(np.mean(np.abs(a - b)))   # MAE across 7 factors
    return {
        "mean_mae": mae(gen.mean(0),        oos.mean(0)),
        "var_mae":  mae(gen.var(0),         oos.var(0)),
        "skew_mae": mae(skew(gen, axis=0),  skew(oos, axis=0)),
        "kurt_mae": mae(kurtosis(gen, 0),   kurtosis(oos, 0)),
        "cov_fro":  float(compute_cov(gen, oos)),
    }
```

### 文件 2：`ablations/run_data_scaling.py`（driver，主体）
```python
# 伪代码
F = pd.read_csv("model/full/factors.csv", index_col=0, parse_dates=True)   # 读一次
BASE_CFG = load_cfg("DDPM")                 # copy eval 的 levy_alpha/timesteps/bs/lr/epochs/use_L_noise/factors
FACTORS  = BASE_CFG["factors"]              # 7 列（丢 alpha）
os.makedirs("model/data_scaling/checkpoints", exist_ok=True)

rows = []
for k in range(1, 21):                      # 20 折
    train_end, test_yr = 2000 + 4 + k, 2005 + k          # (2005,2006) ... (2024,2025)
    train_df = F[F.index.year <= train_end]
    test     = F[F.index.year == test_yr][FACTORS].values
    tag      = f"fold{k:02d}"

    # (a) 切片 csv + cfg —— 复用现有训练脚本
    train_df.to_csv(f"model/data_scaling/factors_{tag}.csv")
    cfg = {**BASE_CFG, "data_file": f"factors_{tag}.csv", "ckpt_name": f"ds_{tag}"}
    yaml.safe_dump(cfg, open("model/data_scaling/cfg.yaml", "w"))
    subprocess.run(["python", "factor_diffusion_train.py", "data_scaling"], check=True)

    # (b) 采样 4096（原始单位）
    ckpt = f"model/data_scaling/checkpoints/ds_{tag}.pt"
    gen  = DiffusionSampler(ckpt).generate(4096)
    sc   = torch.load(ckpt)["scaler"]
    res  = ResampleSampler(train_df[FACTORS], sc).generate(4096)
    gau  = GaussianSampler(train_df[FACTORS], sc).generate(4096)

    # (c) 指标
    rows.append({"fold": k, "train_years": k + 4, "test_year": test_yr,
                 **{f"gen_{m}": v for m, v in evaluate(gen, test).items()},
                 **{f"res_{m}": v for m, v in evaluate(res, test).items()},
                 **{f"gau_{m}": v for m, v in evaluate(gau, test).items()}})

pd.DataFrame(rows).to_csv("ablations/data_scaling_results.csv", index=False)
```
> 同折一起算 Resample/Gaussian，是为了把「测试年难度」归一化（可画 `gen/res` 相对值）——
> 解决 walk-forward 测试集每折变动的混淆。

### 文件 3：`ablations/data_scaling.py`（出表/图，仿 `corr_.py`）
- 读 `data_scaling_results.csv` → 主表（fold × 5 指标，gen 与 res/gau 并列）。
- 主图：5 个指标 vs 训练年数；叠 Resample/Gaussian 参照线，或直接画 `gen/res` 相对值。

### 唯一可能需要碰训练脚本的点：多 seed
现训练脚本不读 seed。**先跑 1 seed 通 20 折**（完全不用改任何 .py）。
若要 seed∈{0,1,2} 补方差，最小改动是给 `factor_diffusion_train.__main__` 加两行
`torch.manual_seed(cfg.get("seed",0)); np.random.seed(...)`，并在 driver 的 cfg/`ckpt_name`
里带 seed —— 这是整套唯一的侵入式改动，可后置。

### 跑前自检
- 列顺序：fold csv 与 `model/full/factors.csv` 同格式；OOS 切片用 `FACTORS` 同序。
- `BASE_CFG` 直接 copy 自当前 eval 用的 cfg（确认 `levy_alpha`、`use_L_noise` 与 PDF 一致）。
- 第一折先单跑一遍，确认 ckpt 生成、`generate(4096)` 返回 (4096,7)、`evaluate` 出 5 个数，再放开 20 折循环。

</details>

---

## 预期结果与解读

- **相对值曲线（gen/Resample）随训练年数下降并走平** → 数据量有用但边际递减；走平点就是「日频数据的饱和长度」。
- **早早饱和（如 10y 就到顶）** → 当前日频数据已饱和，再多历史无益；这正是**支持做分钟级数据集（见 massiveDS/plan.md）的论据**——想继续提升得换更高频/更多样本，而非更长历史。两实验互相呼应。
- **小窗口 Kurt MAE 反而更低** → 警惕假性变好：小数据可能塌缩到低方差解（参考 `corr.md` 里 KS 因过度集中失效），结合 Mahalanobis Q-Q / KDE 判断。

---