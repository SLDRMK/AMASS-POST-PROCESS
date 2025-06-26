import json
import pandas as pd
from pathlib import Path

# 读取JSON文件
json_file = "logs/MotionTracking/20250624_165713-tennis-motion_tracking-g1_23dof_lock_wrist/metrics/ckpt_119200/samtraj.json"

with open(json_file, 'r') as f:
    data = json.load(f)

# 提取数据
raw_data = data['_raw'][0]  # 取第一个样本
accuracy_data = data['accuracy']
smoothness_data = data['smoothness']

# 创建汇总表格
print("=" * 80)
print("运动跟踪任务评估指标汇总 (Tennis Motion Tracking)")
print("=" * 80)
print(f"检查点: ckpt_119200")
print(f"任务: tennis-motion_tracking-g1_23dof_lock_wrist")
print()

# 1. 准确性指标 (Accuracy Metrics)
print("📊 准确性指标 (Accuracy Metrics)")
print("-" * 50)
accuracy_df = pd.DataFrame([
    {
        "指标": "E_gmpjpe",
        "描述": "全局平均关节位置误差 (mm)",
        "数值": accuracy_data['E_gmpjpe']['mean'],
        "标准差": accuracy_data['E_gmpjpe']['std']
    },
    {
        "指标": "E_mpjpe", 
        "描述": "平均关节位置误差 (mm)",
        "数值": accuracy_data['E_mpjpe']['mean'],
        "标准差": accuracy_data['E_mpjpe']['std']
    },
    {
        "指标": "E_dof_mpjpe",
        "描述": "自由度平均位置误差 (mm)",
        "数值": accuracy_data['E_dof_mpjpe']['mean'],
        "标准差": accuracy_data['E_dof_mpjpe']['std']
    },
    {
        "指标": "E_dof_vel",
        "描述": "自由度速度误差 (rad/s)",
        "数值": accuracy_data['E_dof_vel']['mean'],
        "标准差": accuracy_data['E_dof_vel']['std']
    },
    {
        "指标": "E_dof_acc",
        "描述": "自由度加速度误差 (rad/s²)",
        "数值": accuracy_data['E_dof_acc']['mean'],
        "标准差": accuracy_data['E_dof_acc']['std']
    },
    {
        "指标": "E_vel",
        "描述": "根节点速度误差 (m/s)",
        "数值": accuracy_data['E_vel']['mean'],
        "标准差": accuracy_data['E_vel']['std']
    },
    {
        "指标": "E_root_vel",
        "描述": "根节点速度误差 (m/s)",
        "数值": accuracy_data['E_root_vel']['mean'],
        "标准差": accuracy_data['E_root_vel']['std']
    },
    {
        "指标": "E_acc",
        "描述": "根节点加速度误差 (m/s²)",
        "数值": accuracy_data['E_acc']['mean'],
        "标准差": accuracy_data['E_acc']['std']
    },
    {
        "指标": "E_root_acc",
        "描述": "根节点加速度误差 (m/s²)",
        "数值": accuracy_data['E_root_acc']['mean'],
        "标准差": accuracy_data['E_root_acc']['std']
    },
    {
        "指标": "E_contact_acc",
        "描述": "接触加速度误差 (m/s²)",
        "数值": accuracy_data['E_contact_acc']['mean'],
        "标准差": accuracy_data['E_contact_acc']['std']
    }
])

print(accuracy_df.to_string(index=False, float_format='%.2f'))
print()

# 2. 平滑性指标 (Smoothness Metrics)
print("🔄 平滑性指标 (Smoothness Metrics)")
print("-" * 50)
smoothness_df = pd.DataFrame([
    {
        "指标": "L2_vel",
        "描述": "速度L2范数",
        "数值": smoothness_data['L2_vel']['mean'],
        "标准差": smoothness_data['L2_vel']['std']
    },
    {
        "指标": "L2_acc",
        "描述": "加速度L2范数",
        "数值": smoothness_data['L2_acc']['mean'],
        "标准差": smoothness_data['L2_acc']['std']
    },
    {
        "指标": "L2_jerk",
        "描述": "加加速度L2范数",
        "数值": smoothness_data['L2_jerk']['mean'],
        "标准差": smoothness_data['L2_jerk']['std']
    },
    {
        "指标": "L2_dof_vel",
        "描述": "自由度速度L2范数",
        "数值": smoothness_data['L2_dof_vel']['mean'],
        "标准差": smoothness_data['L2_dof_vel']['std']
    },
    {
        "指标": "L2_dof_acc",
        "描述": "自由度加速度L2范数",
        "数值": smoothness_data['L2_dof_acc']['mean'],
        "标准差": smoothness_data['L2_dof_acc']['std']
    },
    {
        "指标": "L2_dof_jerk",
        "描述": "自由度加加速度L2范数",
        "数值": smoothness_data['L2_dof_jerk']['mean'],
        "标准差": smoothness_data['L2_dof_jerk']['std']
    },
    {
        "指标": "L2_ref_vel",
        "描述": "参考速度L2范数",
        "数值": smoothness_data['L2_ref_vel']['mean'],
        "标准差": smoothness_data['L2_ref_vel']['std']
    },
    {
        "指标": "L2_ref_acc",
        "描述": "参考加速度L2范数",
        "数值": smoothness_data['L2_ref_acc']['mean'],
        "标准差": smoothness_data['L2_ref_acc']['std']
    },
    {
        "指标": "L2_ref_jerk",
        "描述": "参考加加速度L2范数",
        "数值": smoothness_data['L2_ref_jerk']['mean'],
        "标准差": smoothness_data['L2_ref_jerk']['std']
    },
    {
        "指标": "L2_ref_dof_vel",
        "描述": "参考自由度速度L2范数",
        "数值": smoothness_data['L2_ref_dof_vel']['mean'],
        "标准差": smoothness_data['L2_ref_dof_vel']['std']
    },
    {
        "指标": "L2_ref_dof_acc",
        "描述": "参考自由度加速度L2范数",
        "数值": smoothness_data['L2_ref_dof_acc']['mean'],
        "标准差": smoothness_data['L2_ref_dof_acc']['std']
    },
    {
        "指标": "L2_ref_dof_jerk",
        "描述": "参考自由度加加速度L2范数",
        "数值": smoothness_data['L2_ref_dof_jerk']['mean'],
        "标准差": smoothness_data['L2_ref_dof_jerk']['std']
    }
])

print(smoothness_df.to_string(index=False, float_format='%.2f'))
print()

# 3. 关键指标总结
print("🎯 关键指标总结")
print("-" * 50)
print(f"• 平均关节位置误差 (MPJPE): {accuracy_data['E_mpjpe']['mean']:.2f} mm")
print(f"• 全局平均关节位置误差 (GMPJPE): {accuracy_data['E_gmpjpe']['mean']:.2f} mm")
print(f"• 根节点速度误差: {accuracy_data['E_root_vel']['mean']:.2f} m/s")
print(f"• 根节点加速度误差: {accuracy_data['E_root_acc']['mean']:.2f} m/s²")
print(f"• 运动平滑性 (速度L2): {smoothness_data['L2_vel']['mean']:.2f}")
print(f"• 运动平滑性 (加速度L2): {smoothness_data['L2_acc']['mean']:.2f}")
print()

# 4. 性能评估
print("📈 性能评估")
print("-" * 50)
mpjpe = accuracy_data['E_mpjpe']['mean']
gmpjpe = accuracy_data['E_gmpjpe']['mean']

if mpjpe < 50:
    mpjpe_rating = "优秀"
elif mpjpe < 100:
    mpjpe_rating = "良好"
elif mpjpe < 150:
    mpjpe_rating = "一般"
else:
    mpjpe_rating = "需要改进"

if gmpjpe < 100:
    gmpjpe_rating = "优秀"
elif gmpjpe < 200:
    gmpjpe_rating = "良好"
elif gmpjpe < 300:
    gmpjpe_rating = "一般"
else:
    gmpjpe_rating = "需要改进"

print(f"• MPJPE ({mpjpe:.2f} mm): {mpjpe_rating}")
print(f"• GMPJPE ({gmpjpe:.2f} mm): {gmpjpe_rating}")
print(f"• 自由度跟踪精度: {accuracy_data['E_dof_mpjpe']['mean']:.2f} mm")
print(f"• 接触稳定性: {accuracy_data['E_contact_acc']['mean']:.2f} m/s²")
print()

print("=" * 80) 