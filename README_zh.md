# PINNNMEoS 代码说明

本仓库分为两个独立阶段：固定优化参数点的 PINN 训练，以及传统 RMF
求解器的参数采样、核物质筛选、EOS/TOV-Love 计算和天文相容性筛选。
核物质性质和天文观测不进入本次 PINN 训练损失。

## 本次严格按新稿调整的设置

- 参数输入除以表 I 原始 FSU-delta 参数的绝对值。
- 总损失采用四个归一化场残差的批次均方之和，不再额外除以四。
- 截断高斯采样标准差为原始参考参数绝对值的40%，中心为优化点。
- 核物质对称能区间按你最后确认的27.7-34.1 MeV执行。
- R2 同时检验 J0740、J0437 的质量-半径和 GW170817 联合质量-潮汐后验。

这些变更与产生旧结果的程序不同。旧的33.76分钟、最终损失、402/296等
数据不能直接作为新配置的输出。旧模型单独标为 historical，程序不会修改它。
本次没有重新执行20万步训练或43.2万组采样。

## 安装与运行

推荐 Python 3.10 及以上，创建虚拟环境后执行：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

训练：

```bash
python scripts/train_fsu_delta_pinn_fields_variable10.py --device cuda --outdir outputs/train
```

GPU训练不受CPU推理的单线程限制。训练输出包括逐步总损失、分项损失、
学习率、采样内存、张量显存峰值和检查点。采样RSS不等于操作系统工作集峰值。

导出新模型并测试单线程CPU速度：

```bash
python scripts/export_checkpoint.py --checkpoint outputs/train/latest.pt --output outputs/model.npz
python scripts/benchmark_field_solver_numpy.py --checkpoint outputs/model.npz --output outputs/benchmark.json --repeats 31
```

生成R1并筛选R2：

```bash
python scripts/wide_r1r2_20261103.py --output outputs/ensemble --bps vendor/rmf/TOVtools/bps.dat --count 432000 --workers 8
python scripts/select_r2.py --bank outputs/ensemble --observations data/observations
```

根据机器资源选择 workers；它表示进程数，不表示实际独占物理核数量。
原始后验的来源、处理约定及数据权利说明见 docs/OBSERVATIONS.md。

## 上传 GitHub 前

上传本目录内容，而不是整个论文工作目录。无需上传 outputs、缓存或虚拟环境。
README.md 是英文首页，CITATION.cff 提供引用信息，.github/workflows 提供CPU测试。
LICENSE 对作者原创代码提供MIT授权；已有工具代码和观测数据需保留其适用权利。
配置差异与尚需确认的事项详见 docs/MANUSCRIPT_ALIGNMENT.md，特别是论文未列出的
截断上下界和旧数据无法自动复现的问题。此包整理完成不等于已经上传或公开发布。
