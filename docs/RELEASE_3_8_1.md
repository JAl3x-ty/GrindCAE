# GrindCAE 3.8.1

## 文档体系

用户已确认高级参数与滚轮修复无问题，人工验收通过，授权Windows便携封包及本地Git提交。
用户明确远程暂不提交。发布目录继续沿用 project-Grinding.CAE-3.8.0；独立本地分支为 codex/release-3.8.1。
详细能力与模型假设见 IMPLEMENTATION_3_8_1.md，数值和人工验收见 ACCEPTANCE_3_8_1.md。

## 开发边界

版本元数据统一3.8.1，保留独立发行包名grindcae-literature-kernel。
集成论文四分量力、工程参数绑定、守恒条带投影、双路线J2历史、卸载残余、结果中心及导出。
修复高级参数布局溢出和当前节点滚轮路由。
源码候选274项回归及人工验收通过。正式版本在独立构建运行时重新回归后提交，再从干净Git状态构建。

封包使用PyInstaller onedir，包含GrindCAE.exe、GrindCAE-Diagnostics.exe、运行库、依赖清单、许可证与快速入门。
构建脚本强制冻结程序文献/J2联算、结果回读及Tk工程往返检查；保留旧模式冻结smoke。
ZIP、SHA256及manifest写入release目录，manifest记录构建源码提交。
独立电脑验证仍需真实执行；开发机便携检查不代表该项已通过。

原project-Grinding.CAE工作区及4.0/Abaqus未提交成果保持原位。
本仓库为已验收并行工作台的独立源码快照；保留的4.0相关代码不属于此次论文路线的验收或新增发布能力。
无远程推送、无论文原PDF上传、无用户计算目录入库。

## 2026-09-25 封包结果

- 用户最终人工验收已通过；仅本地Git，不推送远程。
- 构建源码提交：`4213ee3adc15823ae5718c6db3ba2f824e71a2c7`，构建前工作树干净。
- 独立构建运行时按测试文件分进程执行20个文件，274项全部通过，见 evidence/381_isolated_tests.json。
  单进程大批Tk窗口曾出现tk.tcl主题载入错误，保留失败日志；分进程检查未删减项目。
- PyInstaller构建通过；冻结运行库诊断、单位置、单程扫描、机制场演化及文献/J2联算通过。
- 冻结文献结果：Fn=65.541382 N、Ft=43.817306 N；分项投影最大残差7.11e-15 N；两路线最终零外载；Tk工作台和工程保存回读通过。
- ZIP：release/GrindCAE-3.8.1-Windows-x64.zip，130612569 bytes；1430个ZIP条目CRC检查通过。
- SHA256：`b99617e5d405ccdbcad9cb734d4e86332f9578e62c1094615e53599ef25eb245`。
- 证据：evidence/381_release_manifest.json、381_frozen_literature_smoke.json、381_frozen_legacy_smoke.json。
- 双击解压后的GrindCAE.exe；源码启动器继续保留。便携包保存在release目录，不将二进制运行库加入源码Git。
- 独立电脑验收及公开分发许可审查仍为pending；本机封包检查已完成，独立物理验证未完成。

