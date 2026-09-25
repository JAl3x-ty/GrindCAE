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
