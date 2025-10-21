# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 14:35:07 2025

@author: ypan1
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# 设置页面
st.set_page_config(
    page_title="稳健统计分析工具",
    page_icon="📊",
    layout="wide"
)

# 标题和说明
st.title("📊 稳健统计分析工具")
st.markdown("""
基于算法A的稳健统计分析方法，用于处理包含异常值的数据集。
支持数据上传、自动分析、可视化展示和结果导出。
""")

# 侧边栏 - 参数设置
st.sidebar.header("⚙️ 分析参数")
k_value = st.sidebar.slider("尺度因子 (k)", 1.0, 3.0, 1.5, 0.1)
max_iter = st.sidebar.slider("最大迭代次数", 10, 100, 50)

# 数据输入方式选择
input_method = st.radio("数据输入方式:", 
                       ["手动输入", "文件上传", "示例数据"])

data = None

if input_method == "手动输入":
    st.subheader("📝 手动输入数据")
    data_input = st.text_area("请输入数据（每行一个数值或用逗号分隔）:", 
                             "4.7, 4.6, 4.66, 4.7, 4.7, 4.93, 4.65")
    
    if st.button("分析数据"):
        try:
            # 解析输入数据
            if "\n" in data_input:
                data_list = [float(x.strip()) for x in data_input.split("\n") if x.strip()]
            else:
                data_list = [float(x.strip()) for x in data_input.split(",") if x.strip()]
            
            data = np.array(data_list)
            st.success(f"成功解析 {len(data)} 个数据点")
            
        except ValueError as e:
            st.error("数据格式错误！请确保输入的是数字")

elif input_method == "文件上传":
    st.subheader("📁 上传数据文件")
    uploaded_file = st.file_uploader("选择CSV或TXT文件", type=['csv', 'txt'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                # 假设第一列是数据
                data = df.iloc[:, 0].values
            else:
                # 文本文件，每行一个数字
                content = uploaded_file.read().decode()
                data_list = [float(x.strip()) for x in content.split() if x.strip()]
                data = np.array(data_list)
            
            st.success(f"成功加载 {len(data)} 个数据点")
            st.write("前10个数据:", data[:10])
            
        except Exception as e:
            st.error(f"文件读取错误: {e}")

else:  # 示例数据
    st.subheader("🎯 示例数据分析")
    # 使用您之前的数据作为示例
    example_data = np.array([
        4.7, 4.6, 4.66, 4.7, 4.7, 4.93, 4.65, 4.61, 4.73, 4.6,
        4.7, 4.88, 4.59, 4.8, 4.8, 4.68, 4.73, 4.35, 4.78, 4.71,
        4.72, 4.72, 4.63, 4.65, 4.66, 4.64, 4.38, 4.4, 4.64, 4.63,
        4.6, 4.25, 4.53, 4.34, 4.16, 4.26, 4.68, 4.74, 4.3, 4.4,
        4.32, 4.34, 4.38, 4.38, 4.27, 4.38, 4.81, 4.37, 4.39, 4.38,
        4.39, 4.34, 4.33, 4.35, 3.85, 4.31, 4.52, 4.73, 3.95, 3.83,
        4.37, 4.27, 3.71, 4.11, 4.37
    ])
    data = example_data
    st.write("示例数据已加载，包含65个测量值")

# 稳健统计算法函数
def robust_algorithm_a(data, max_iterations=50, k=1.5):
    """算法A稳健统计实现"""
    n = len(data)
    
    # 初始值
    X_star = np.median(data)
    abs_deviations = np.abs(data - X_star)
    median_abs_deviation = np.median(abs_deviations)
    S_star = 1.483 * median_abs_deviation
    
    # 迭代过程
    converged = False
    iteration = 0
    history = []
    
    while iteration < max_iterations and not converged:
        iteration += 1
        prev_X_star = X_star
        prev_S_star = S_star
        
        # 计算δ并修正数据点
        delta = k * S_star
        Xj_star = np.where(data < X_star - delta, X_star - delta, 
                          np.where(data > X_star + delta, X_star + delta, data))
        
        # 重新计算
        X_star = np.mean(Xj_star)
        sum_squared_deviations = np.sum((Xj_star - X_star)**2)
        S_star = 1.134 * np.sqrt(sum_squared_deviations / (n-1))
        
        # 记录历史
        history.append({
            'iteration': iteration,
            'X_star': X_star,
            'S_star': S_star,
            'delta': delta
        })
        
        # 检查收敛
        if (int(prev_X_star * 1000) == int(X_star * 1000) and 
            int(prev_S_star * 1000) == int(S_star * 1000)):
            converged = True
    
    # 最终结果
    final_delta = k * S_star
    lower_limit = X_star - final_delta
    upper_limit = X_star + final_delta
    outliers_mask = (data < lower_limit) | (data > upper_limit)
    outliers = data[outliers_mask]
    clean_data = data[~outliers_mask]
    Z_scores = (data - X_star) / S_star
    
    return {
        'robust_mean': X_star,
        'robust_std': S_star,
        'clean_data': clean_data,
        'outliers': outliers,
        'Z_scores': Z_scores,
        'iterations': iteration,
        'converged': converged,
        'lower_limit': lower_limit,
        'upper_limit': upper_limit,
        'history': history
    }

# 执行分析
if data is not None and len(data) > 0:
    st.markdown("---")
    st.subheader("📈 分析结果")
    
    with st.spinner("正在执行稳健统计分析..."):
        results = robust_algorithm_a(data, max_iterations=max_iter, k=k_value)
    
    # 创建两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("稳健均值 (X*)", f"{results['robust_mean']:.6f}")
        st.metric("稳健标准差 (S*)", f"{results['robust_std']:.6f}")
        
    with col2:
        st.metric("迭代次数", results['iterations'])
        st.metric("离群值数量", len(results['outliers']))
    
    # 详细结果
    st.subheader("📋 详细结果")
    
    st.write(f"**正常值范围**: [{results['lower_limit']:.6f}, {results['upper_limit']:.6f}]")
    st.write(f"**收敛状态**: {'是' if results['converged'] else '否'}")
    
    if len(results['outliers']) > 0:
        st.write(f"**离群值**: {results['outliers']}")
    else:
        st.write("**离群值**: 无")
    
    # Z比分数统计
    z_scores_abs = np.abs(results['Z_scores'])
    satisfactory = np.sum(z_scores_abs <= 2)
    questionable = np.sum((z_scores_abs > 2) & (z_scores_abs <= 3))
    unsatisfactory = np.sum(z_scores_abs > 3)
    
    st.write("**Z比分数分类**:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("满意 (|Z| ≤ 2)", f"{satisfactory} 个")
    with col2:
        st.metric("可疑 (2 < |Z| ≤ 3)", f"{questionable} 个")
    with col3:
        st.metric("不满意 (|Z| > 3)", f"{unsatisfactory} 个")
    
    # 可视化
    st.subheader("📊 数据可视化")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 子图1: 数据分布
    ax1.hist(data, bins=20, alpha=0.7, color='skyblue', edgecolor='black', label='所有数据')
    ax1.hist(results['clean_data'], bins=15, alpha=0.7, color='lightgreen', 
             edgecolor='black', label='正常数据')
    ax1.axvline(results['robust_mean'], color='red', linestyle='--', 
                label=f'稳健均值: {results["robust_mean"]:.4f}')
    ax1.axvline(results['lower_limit'], color='orange', linestyle=':')
    ax1.axvline(results['upper_limit'], color='orange', linestyle=':')
    ax1.set_xlabel('数值')
    ax1.set_ylabel('频数')
    ax1.set_title('数据分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: Z比分数
    ax2.hist(results['Z_scores'], bins=15, alpha=0.7, color='lightcoral', 
             edgecolor='black')
    for z in [-3, -2, 0, 2, 3]:
        color = 'red' if abs(z) == 3 else 'orange' if abs(z) == 2 else 'red'
        linestyle = '-' if z == 0 else '--'
        ax2.axvline(z, color=color, linestyle=linestyle, alpha=0.7)
    ax2.set_xlabel('Z比分数')
    ax2.set_ylabel('频数')
    ax2.set_title('Z比分数分布')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 箱线图
    box_data = [data, results['clean_data']]
    box_plot = ax3.boxplot(box_data, labels=['原始数据', '清洗后数据'], 
                          patch_artist=True)
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    ax3.set_ylabel('数值')
    ax3.set_title('数据分布比较')
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 迭代过程
    iterations = [h['iteration'] for h in results['history']]
    X_stars = [h['X_star'] for h in results['history']]
    S_stars = [h['S_star'] for h in results['history']]
    
    ax4.plot(iterations, X_stars, 'o-', label='稳健均值 (X*)')
    ax4.plot(iterations, S_stars, 's-', label='稳健标准差 (S*)')
    ax4.set_xlabel('迭代次数')
    ax4.set_ylabel('数值')
    ax4.set_title('迭代收敛过程')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 导出功能
    st.subheader("💾 导出结果")
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        '原始数据': data,
        'Z比分数': results['Z_scores'],
        '分类': np.where(np.abs(results['Z_scores']) <= 2, '满意',
                       np.where(np.abs(results['Z_scores']) <= 3, '可疑', '不满意'))
    })
    
    # 下载CSV
    csv = result_df.to_csv(index=False)
    st.download_button(
        label="下载完整结果CSV",
        data=csv,
        file_name="稳健分析结果.csv",
        mime="text/csv"
    )
    
    # 下载报告
    report = f"""
稳健统计分析报告
================

分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
数据点数: {len(data)}
尺度因子(k): {k_value}

关键结果:
--------
稳健均值 (X*): {results['robust_mean']:.6f}
稳健标准差 (S*): {results['robust_std']:.6f}
正常值范围: [{results['lower_limit']:.6f}, {results['upper_limit']:.6f}]
迭代次数: {results['iterations']}
离群值数量: {len(results['outliers'])}

数据质量分类:
-----------
满意 (|Z| ≤ 2): {satisfactory} 个数据点
可疑 (2 < |Z| ≤ 3): {questionable} 个数据点  
不满意 (|Z| > 3): {unsatisfactory} 个数据点
"""
    
    st.download_button(
        label="下载分析报告",
        data=report,
        file_name="稳健分析报告.txt",
        mime="text/plain"
    )

else:
    st.info("👆 请先输入或上传数据以开始分析")

# 页脚
st.markdown("---")
st.markdown("*基于GB/T 28043-2019/ISO 13528:2015 算法A的稳健统计分析方法*")