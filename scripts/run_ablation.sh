#!/bin/bash

# 确保脚本遇到错误时中断，保护 API Token 余额
set -e

echo "🚀 [Ablation Pipeline] 开始执行全量消融实验方案..."
echo "================================================================="

# 1. 运行完整框架模型 (Ours)
# echo "💎 [1/4] 正在启动 Full Framework (Ours) 实验..."
# python enhanced_lhpr_vln_runner.py --cfg_file ours
# echo "✅ Ours 实验运行完毕！"
# echo "-----------------------------------------------------------------"

# 2. 运行无长时记忆消融环境 (w/o Memory)
echo "🧠 [2/4] 正在启动 w/o Memory 消融实验..."
python scripts/Ablation_study_runnner.py --cfg_file wo_memory
echo "✅ w/o Memory 实验运行完毕！"
echo "-----------------------------------------------------------------"

# 3. 运行无前沿打分消融环境 (w/o Frontier Score)
echo "🎯 [3/4] 正在启动 w/o Frontier Score 消融实验..."
python scripts/Ablation_study_runnner.py --cfg_file wo_frontier_score
echo "✅ w/o Frontier Score 实验运行完毕！"
echo "-----------------------------------------------------------------"

# 4. 运行无安全过滤器消融环境 (w/o Safety Filter)
echo "🚨 [4/4] 正在启动 w/o Safety Filter 消融实验..."
python scripts/Ablation_study_runnner.py --cfg_file wo_safety_filter
echo "✅ w/o Safety Filter 实验运行完毕！"

echo "================================================================="
echo "🎉 [Ablation Complete] 所有消融实验全部安全完结！数据已成功存盘。"

# 5. 运行无场景图拓扑消融环境 (w/o Scene Graph)
echo "🌐 [5/5] 正在启动 w/o Scene Graph 消融实验..."
python scripts/Ablation_study_runnner.py --cfg_file wo_scene_graph
echo "✅ w/o Scene Graph 实验运行完毕！"

# 6. 运行无场景图拓扑消融环境 (baseline)
echo "🌐 [5/5] 正在启动 baseline 消融实验..."
python scripts/Ablation_study_runnner.py --cfg_file baseline
echo "✅ baseline 实验运行完毕