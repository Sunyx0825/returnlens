# ReturnLens

**美股月度超额收益预测：把回归约束放到真正未参与估计的数据上检验。**

股息、估值和利率指标能否预测下个月的市场超额收益？ReturnLens用15个单变量回归回答这个问题，并比较加入系数符号约束、非负预测截断后，预测误差是否下降。

项目基于Goyal–Welch的研究框架。我的工作包括Python预测流程、训练期符号约束、信息时点处理，以及统一的样本外比较。约束思路与Campbell–Thompson的文献相关；本项目采用初始训练期估计的方向，并明确保留原回归截距。

## 结果

使用1927—2005年月度数据，以20年历史作为初始估计窗口，每月增加一个观测并预测下一月。所有比较使用同一组测试月份。

| 测试集指标 | 结果 |
|---|---:|
| 约束后样本外R²提高 | **12/15个模型** |
| 预测误差低于历史均值的模型 | **4个→8个** |
| 表现下降 / 不变 | 2个 / 1个 |

这里的R²衡量相对历史均值预测的误差改善，不代表交易收益。完整结果包括表现未改善的模型，见[结果表](results/comparison.csv)。

## 快速运行

需要Python3.11或更新版本。在仓库目录执行：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m returnlens.cli
python -m pytest
```

仓库包含本次研究使用的月度数据快照，无需API密钥。程序在`output/`生成：

- `comparison.csv`：15个模型的前后对比。
- `forecasts.csv`：每个预测时点的输入日期、训练样本量、系数、预测值和实际收益。
- `metrics.csv`：测试月份、均方误差及样本外R²。
- `summary.json`：汇总结果、数据哈希和运行环境。

也可以使用自己的同格式数据：

```bash
python -m returnlens.cli --data path/to/monthly.csv --output output/custom
```

## 如何避免提前使用未来信息

1. 在月份`t`结束时，只用截至`t`的已知收益估计回归，预测`t+1`。
2. 系数方向只在初始训练窗口估计一次；后续若斜率方向冲突，将斜率置零，再把负预测值截为零。
3. 通胀数据多等待一个月，训练和预测采用一致的滞后。
4. 历史均值基准也逐月更新，与回归模型使用相同的信息截止时间。

测试会修改未来数据，检查更早的预测是否保持不变。完整变量定义、时间窗口及指标公式见[研究方法](docs/methodology.md)。

## 阅读代码

```text
src/returnlens/
  data.py        数据检查与指标构造
  forecast.py    扩展窗口估计与预测约束
  evaluate.py    同期误差比较
  cli.py         研究运行入口
tests/           时点隔离、公式与输入检查
results/         已复算的研究结果
```

## 数据与参考文献

数据为原项目保存的Goyal–Welch2005年末月度快照，来源与文件校验值见[data/README.md](data/README.md)。

- Goyal, A. & Welch, I. (2008). [A Comprehensive Look at the Empirical Performance of Equity Premium Prediction](https://www.ivo-welch.info/research/journalcopy/2008-rfs.pdf).
- Campbell, J. Y. & Thompson, S. B. (2008). [Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?](https://doi.org/10.1093/rfs/hhm055).
- [原始年度与月度复现项目](https://github.com/Sunyx0825/Reproduction-of-Goyal-Welch-2008-)

维护者：[Yixuan Sun](https://github.com/Sunyx0825)。问题和复现反馈可以提交到Issues。
