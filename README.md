# GrindCAE 3.8.1 文献力与弹塑性历史工作台

## 文档体系

当前活动任务为 docs/TASK_3_8_1.md；能力与验收见 [3.8.1验收说明](docs/ACCEPTANCE_3_8_1.md)。
最新状态见 docs/PROJECT_STATE.md，以下3.8.0内容为历史记录。

## 开发边界

新增论文力驱动的均匀/周向条带双路线J2历史联算，已执行自动及GUI核验。
用户已确认入口、结果、看图正常；随后授权的高级参数布局与滚轮修复已完成。
启动 Start-GrindCAE-3.8.1.cmd；旧窗口需要用户保存后重开才能载入修复。
用户已通过人工验收并授权封包。目录沿用3.8.0名称，原4.0/Abaqus树保留。发布状态见 docs/RELEASE_3_8_1.md。
模型补齐假设、材料标定及独立实验验证边界见验收说明。

## 历史记录

## 文档体系

本目录为独立的3.8.0开发候选，依据Zhang等2017年材料去除与塑性堆积磨削力模型。
完整科学阅读、公式、单位、歧义与复现缺口见 [论文审计](docs/PAPER_AUDIT.md)。
实施范围见 [任务书](docs/TASK.md)，当前验收状态见 [项目状态](docs/PROJECT_STATE.md)。

## 开发边界

### 当前候选：3.8.0rc2 现有工程工作台集成

已完成第七种正式分析模式“文献磨削力·材料去除与塑性堆积”，复用参数保存、后台计算、
结果中心、Origin导出及工程归档。179项检查通过（29.73s），实际Tk计算和界面截图已检查。
双击Start-GrindCAE-3.8.0.cmd启动完整工作台；在分析页选择文献模式。
详见[集成候选说明](docs/RELEASE_3_8_0_RC2.md)。原工作区385个源码/测试哈希保持一致。
副本版本3.8.0rc2；原工作区仍3.0.1，保留真实接触开发。当前停在用户人工试用。
FE空间载荷/历史联算和便携EXE仍未交付。下方rc1/dev0为历史记录，以本条为当前状态。

### 历史候选：3.8.0rc1 中文桌面工作台

**双击根目录 `Start-GrindCAE-3.8.0.cmd` 启动。** 使用本机现有Python环境。
支持参数编辑、算例保存加载、后台计算、分项图/磨粒表、历史结果和现有内核总力输入导出。
46项测试通过，9份历史结果回读成功，实际窗口已检查；待用户人工试用。
使用步骤和范围见[候选版说明](docs/RELEASE_3_8_0_RC1.md)。
完整FE历史/主工作台合并及便携EXE尚未完成。以下dev0段落保留为开发历史。

2026-09-25追加：合理补齐版及现有内核总力接口已实现，详见
[总体升级设计](docs/INTEGRATED_KERNEL_UPGRADE.md)与[实际结果](docs/RECONSTRUCTION_OUTCOME.md)。
新Schema 2支持有界修整高度、两种动态边界和K1参考反标；旧Schema 1保持兼容。
本轮未运行自动测试，版本仍3.8.0.dev0。完整FE衔接和GUI尚未完成。

新示例与图表在`outputs/completed_model_study`，可运行：

```powershell
cd D:\CODEX\project-Grinding.CAE-3.8.0
.\run_kernel.ps1 -Case .\outputs\completed_model_study\bounded_reference_calibrated.json -OutputDirectory .\outputs\my_bounded_run
.\run_legacy_bridge.ps1 -SourceResult .\outputs\my_bounded_run -OutputDirectory .\outputs\my_force_bridge
```

`run_legacy_bridge.ps1`读取现有主内核并调用其公共力API；不会运行FE。
可选`-HistoryTemplate`和`-MaterialProvenance`生成严格的历史输入；必须明确保留E/ν/Et的来源。
旧内核输入里的等效us和Fn/Ft只适用于该工况；修改工艺后应重算文献模型并重新生成。
示例历史输入与来源清单已保存于`outputs/legacy_point_bridge`。

### 首轮内核及旧示例说明

已经实现独立SI输入、β经验曲线、逐磨粒耕犁/切削与摩擦力、随机砂轮字面重建、
动态遮挡、合力、K1双分量反标接口、命令行和严格结果回读。
结果保存完整公式约定、单位假设、种子、逐磨粒深度/力及文件哈希。
现有真实接触源码、检查点和应用3.0.1保留在 `D:/CODEX/project-Grinding.CAE`。
本版没有集成旧GUI，也没有生成便携EXE。原文图15/18精确复现和独立实验验证尚未完成。

### 使用

在PowerShell执行，输出目录须为新目录：

```powershell
cd D:\CODEX\project-Grinding.CAE-3.8.0
.\run_kernel.ps1 -Case .\examples\explicit_depths.json -OutputDirectory .\outputs\my_explicit_run
.\run_kernel.ps1 -Case .\examples\paper_stochastic.json -OutputDirectory .\outputs\my_stochastic_run
```

脚本复用现有Python依赖，使用独立`grindcae380`命名空间，不安装或替换原软件。
另一台机器可用 `-Python` 指定已含NumPy、Matplotlib和Pillow的Python 3.11+。

`explicit_depths`用于提供明确的有效单磨粒切深并核对力公式；所附深度集合是数值示例，不是论文原始砂轮。
`paper_stochastic`按式40字面累计随机振动，带已登记的边界假设；其结果目前明显偏离文献有效磨粒数量。
请先读审计报告，再将其用于研究比较。

输出 `input.json`、`summary.json`、`grains.csv`、`force_components.png`。
CSV用UTF-8 BOM方便Windows导入，计算字段全为SI。结果目录存在时拒绝覆盖。
保存失败不发布半套结果；回读核对输入、文件哈希、逐磨粒数据、力和PNG。

### API

```python
from grindcae380.core import Case, predict, grain_force, calibrate_wear
from grindcae380.workflow import run_case, read_result

case = Case.paper_example(source='explicit_depths')
prediction = predict(case)
```

显式 `formula_convention`：`printed`保留式10及式21最右端，
`continuous_projection`按应力连续性和摩擦投影重建。两者均不能冒称已获作者确认。
参数 `wear_coefficient_N_m`必须配合`wear_provenance`；示例0.01836N·m来自未确认的N·mm单位假设。
`calibrate_wear`分别从实测Fn和Ft反算K1并报告不一致度，不自动平均或替换输入。
材料名固定440C文献拟合；其它材料需要新的β及强度/摩擦依据。

### 核验

```powershell
cd D:\CODEX\project-Grinding.CAE-3.8.0
$env:MPLBACKEND = 'Agg'
& D:\CODEX\project-Grinding.CAE\.venv\Scripts\python.exe -m pytest -q
```

测试使用独立数值积分、手算力、明确遮挡排列、SI单位链、固定种子重复、非法输入拒绝和真实CLI/结果回读。
结果只证明所声明公式合同与实现一致。论文报告的4.19%/4.31%误差不属于本软件。
