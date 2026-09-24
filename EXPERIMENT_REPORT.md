# 1D 波形 DCGAN 实验报告

## 1. 目标与结论

目标是在保持生成波形幅度统计量接近训练数据的前提下，提高相邻采样点的相关性（滞后 1 自相关）。最终选中第 18 轮模型，并已提升为默认检查点。

- 最终模型：`D:\Project_Github\DCGAN-PyTorch\model\model_wave_final.pth`
- 复现训练脚本：`D:\Project_Github\DCGAN-PyTorch\train_wave.py`
- 生成波形网格：`D:\Project_Github\DCGAN-PyTorch\model\generated_waveforms_grid_epoch_18.png`

在 1,024 个固定随机噪声样本、`eval()` 推理模式下，滞后 1 自相关从无辅助约束基线的 **0.370200** 提升为 **0.489598**；真实训练数据为 **0.483918**。最终模型相对真实数据的相关性误差为 **1.17%**。

## 2. 数据与评估协议

- 数据：`D:\Project_Github\DCGAN-PyTorch\data\wav` 下的 18,418 个 WAV 片段。
- 预处理：读取第一声道，统一映射到 `[-1, 1]`；每段长度为 128，末段不足时以零填充。
- 评估：以训练数据全体为参考，固定种子 `369`，从最终生成器采样 1,024 段长度为 128 的波形，并在 `eval()` 模式下计算统计量。
- 选择规则：优先最小化相邻点相关性误差，同时要求标准差、99% 绝对幅度和平均峰值不明显偏离真实数据。

本报告反映训练数据分布相似性，不构成对独立测试集、物理设备或下游任务效果的验证。

## 3. 调参过程

| 试验 | 主要设置 | 观察结果 | 结论 |
| --- | --- | --- | --- |
| 判别器过强试验 | `ndf=64`，`lr_D=2e-4`，`lr_G=2e-4` | 第 2 轮时 `D(G(z))` 约为 0.015，生成器损失升至 4.61 | 拒绝：判别器过快压制生成器 |
| 无辅助约束基线 | `ndf=32`，`lr_D=1e-5`，`lr_G=5e-5`，第 15 轮 | 幅度统计量较好，但滞后 1 自相关仅为 0.370200 | 作为基线保留 |
| 仅相关性约束 | 训练态 `lambda_lag1=10` | 部分检查点相关性接近真实值，但振幅显著偏大；训练态与推理态统计不一致 | 拒绝：不能兼顾推理态幅度与相关性 |
| 推理态联合约束 | `lambda_lag1=5`，`lambda_std=50`，在生成器 `eval()` 输出上计算约束 | 第 18 轮同时达到相关性和幅度统计量要求 | 采用 |

最终复现配置为：`batch_size=256`、`nz=100`、`ngf=64`、`ndf=32`、`lr_D=1e-5`、`lr_G=5e-5`、`beta1=0.5`、18 轮训练。

## 4. 相比原始 DCGAN 的修改

比较基准是仓库 `HEAD` 中的 `dcgan.py` 和 `train.py`。原始实现面向 CelebA 的 64×64 RGB 图像，使用 2D 卷积和单一学习率的 BCE-GAN。

| 模块 | 原始 DCGAN | 波形实现及修改 |
| --- | --- | --- |
| 数据 | CelebA 图像，3 通道、64×64 | WAV 第一声道，归一化后切为 128 点的一维片段 |
| 生成器 | `ConvTranspose2d`，噪声形状为 `(N, nz, 1, 1)` | 全连接层将 `(N, nz)` 映射为 `(N, 4*ngf, 16)`，再经 3 层 `ConvTranspose1d` 生成 `(N, 1, 128)` |
| 判别器 | 5 层 `Conv2d` | 4 层 `Conv1d`，加入 `Dropout(0.3)`；特征宽度从 64 调为 32，避免判别器过强 |
| 优化器 | 生成器与判别器共用 `lr=2e-4` | 使用 `lr_D=1e-5`、`lr_G=5e-5` 的不对称学习率 |
| 生成器损失 | `BCE(D(G(z)), 1)` | 保留 BCE 对抗损失，并增加推理态相关性和标准差约束 |
| 检查点 | 固定周期保存 | 每轮保存至独立实验目录，并按实际 `eval()` 指标选择最佳轮次 |

最终生成器损失为：

```text
L_G = BCE(D(G(z)), 1)
    + 5 × (ρ₁(G_eval(z)) - ρ₁(x_real))²
    + 50 × (std(G_eval(z)) - std(x_real))²
```

其中 `ρ₁(x) = mean(x[t] × x[t+1]) / mean(x[t]²)`，`G_eval` 表示处于 `eval()` 模式的生成器输出。这一设计使优化目标与实际生成时的 BatchNorm 行为一致。

## 5. 最终结果

| 指标 | 真实数据 | 无辅助约束基线（第 15 轮） | 最终模型（第 18 轮） |
| --- | ---: | ---: | ---: |
| 标准差 | 0.013225 | 0.011337 | 0.012284 |
| 99% 绝对幅度 | 0.052734 | 0.052511 | 0.057402 |
| 平均峰值 | 0.051675 | 0.051156 | 0.065986 |
| 滞后 1 自相关 | 0.483918 | 0.370200 | 0.489598 |
| 归一化频谱总变差距离（越低越好） | — | 0.168762 | 0.135193 |

最终模型将相关性绝对误差从 0.113718 降至 0.005680，同时频谱距离降低。代价是平均峰值比真实数据高约 27.7%，因此后续若需更严格的幅度一致性，可继续细调 `lambda_std` 或引入峰值/频谱约束。

## 6. 复现与调用

1. 在 `D:\Project_Github\DCGAN-PyTorch` 目录运行 `python train_wave.py`。
2. 脚本将把复现实验检查点写入 `D:\Project_Github\DCGAN-PyTorch\model\correlation_optimized`。
3. 默认最佳权重已位于 `D:\Project_Github\DCGAN-PyTorch\model\model_wave_final.pth`，可由现有生成代码直接加载。

运行环境为 Python 3.14、PyTorch 2.14.0 CPU。固定了 Python 与 PyTorch 的随机种子，但未启用跨平台确定性算法；不同 PyTorch 版本或硬件可能造成数值轻微变化。




# 多分辨率 STFT 损失实验

本实验沿用 `wavegan_improve` 的 WaveGAN-GP 模型和评估指标，仅在生成器目标中加入多分辨率 STFT 损失：

`G_loss = WGAN_G_loss + 2.5 * MR_STFT_loss`

STFT 参数为 `FFT=[32, 64, 128]`、`hop=[8, 16, 32]`。

在仓库根目录运行训练：

```powershell
python .\wave_improve_dominant_frequency\train_wavegan_torch.py --device cuda --output-dir .\wave_improve_dominant_frequency\checkpoints\mr_stft_2.5 --log-file .\wave_improve_dominant_frequency\log\mr_stft_2.5_epochs.jsonl
```

训练结束后评估主频 Wasserstein 距离和平均 PSD 误差：

```powershell
python .\wave_improve_dominant_frequency\evaluate_wavegan_torch.py --checkpoint .\wave_improve_dominant_frequency\checkpoints\mr_stft_2.5\wavegan_epoch_0200.pt --device cuda --output-dir .\wave_improve_dominant_frequency\images\mr_stft_2.5 --metrics-output .\wave_improve_dominant_frequency\log\mr_stft_2.5_metrics.json
```

权重对比实验依次训练和评估 `0.1, 0.2, ..., 1.0`，为每个权重分别保存 checkpoint、训练日志、评估日志、JSON 指标和评估图，并生成汇总图：

```powershell
python .\wave_improve_dominant_frequency\run_mr_stft_loss_sweep.py --device cuda
```
实验结果不理想,生成的波形愈发扁平