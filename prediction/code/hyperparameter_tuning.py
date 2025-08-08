from config import *
from models import create_model
import itertools

def grid_search_hyperparameters(X, y_bs, quality_labels, ages, feature_dim, 
                               n_folds=3, max_epochs=30, patience=10):
    """
    使用网格搜索寻找最佳超参数
    """
    print("=== 开始网格搜索超参数优化 ===")
    
    # 为了减少搜索空间，选择关键参数进行搜索
    reduced_param_grid = {
        'lstm_units_1': [128, 256],
        'lstm_units_2': [64, 128],
        'dropout_rate': [0.2, 0.3],
        'learning_rate': [1e-4, 1e-3],
        'feature_reduction': [1024, 2048]
    }
    
    print(f"搜索空间包含 {len(list(itertools.product(*reduced_param_grid.values())))} 种组合")
    
    # 存储所有结果
    all_results = []
    best_score = float('inf')  # 对于MAE，越小越好
    best_params = None
    
    # 创建参数组合
    param_combinations = list(itertools.product(*reduced_param_grid.values()))
    param_names = list(reduced_param_grid.keys())
    
    # 使用分层K折交叉验证
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for i, param_values in enumerate(param_combinations):
        # 构建参数字典
        current_params = dict(zip(param_names, param_values))
        
        print(f"\n--- 测试参数组合 {i+1}/{len(param_combinations)} ---")
        print(f"参数: {current_params}")
        
        # 存储当前参数组合的交叉验证结果
        cv_scores = []
        
        # 交叉验证
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, quality_labels)):
            print(f"  折 {fold_idx + 1}/{n_folds}")
            
            # 清理GPU内存
            tf.keras.backend.clear_session()
            gc.collect()
            
            # 拆分数据
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_bs[train_idx], y_bs[val_idx]
            
            try:
                # 创建模型
                model = create_model(feature_dim, **current_params)
                
                # 回调函数
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=patience, 
                                 restore_best_weights=True, verbose=0),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                     patience=patience//2, min_lr=1e-7, verbose=0)
                ]
                
                # 数据管道
                batch_size = 8  # 使用较小的批量大小
                train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                train_ds = train_ds.shuffle(256).batch(batch_size).prefetch(tf.data.AUTOTUNE)
                
                val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
                val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
                
                # 训练模型
                history = model.fit(
                    train_ds, validation_data=val_ds,
                    epochs=max_epochs, callbacks=callbacks, verbose=0
                )
                
                # 获取最佳验证MAE
                best_val_mae = min(history.history['val_mae'])
                cv_scores.append(best_val_mae)
                
                print(f"    验证MAE: {best_val_mae:.4f}")
                
                # 清理内存
                del model, train_ds, val_ds
                tf.keras.backend.clear_session()
                gc.collect()
                
            except Exception as e:
                print(f"    错误: {e}")
                cv_scores.append(float('inf'))
                continue
        
        # 计算平均分数
        if cv_scores and not all(score == float('inf') for score in cv_scores):
            avg_score = np.mean([s for s in cv_scores if s != float('inf')])
            current_result = {
                'params': current_params,
                'cv_scores': cv_scores,
                'avg_score': avg_score,
                'std_score': np.std([s for s in cv_scores if s != float('inf')])
            }
            all_results.append(current_result)
            
            print(f"  平均验证MAE: {avg_score:.4f} ± {current_result['std_score']:.4f}")
            
            # 更新最佳参数
            if avg_score < best_score:
                best_score = avg_score
                best_params = current_params.copy()
                print(f"  *** 新的最佳参数! ***")
        else:
            print(f"  所有折都失败，跳过此参数组合")
    
    # 打印最佳结果
    print(f"\n=== 网格搜索完成 ===")
    print(f"最佳参数: {best_params}")
    print(f"最佳平均验证MAE: {best_score:.4f}")
    
    # 保存结果
    results_summary = {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': all_results,
        'param_grid': reduced_param_grid
    }
    
    with open(OUTPUT_PATHS['grid_search_results'], 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"网格搜索结果已保存至: {OUTPUT_PATHS['grid_search_results']}")
    
    return best_params, best_score, all_results

def plot_grid_search_results(all_results, save_path=None):
    """可视化网格搜索结果"""
    if save_path is None:
        save_path = OUTPUT_PATHS['grid_search_analysis']
        
    if not all_results:
        print("没有结果可可视化")
        return
    
    # 提取数据
    scores = [result['avg_score'] for result in all_results]
    params_list = [result['params'] for result in all_results]
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 分数分布
    axes[0, 0].hist(scores, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(min(scores), color='red', linestyle='--', label=f'Best: {min(scores):.4f}')
    axes[0, 0].set_xlabel('Average Validation MAE')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Cross-Validation Scores')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. 参数重要性分析（简化版）
    param_importance = {}
    for param_name in ['lstm_units_1', 'lstm_units_2', 'dropout_rate', 'learning_rate']:
        if param_name in params_list[0]:
            values = [params[param_name] for params in params_list]
            unique_values = list(set(values))
            
            # 计算每个参数值的平均分数
            param_scores = {}
            for value in unique_values:
                value_scores = [scores[i] for i, params in enumerate(params_list) 
                              if params[param_name] == value]
                param_scores[value] = np.mean(value_scores)
            
            param_importance[param_name] = param_scores
    
    # 绘制参数重要性
    if param_importance:
        param_names = list(param_importance.keys())
        n_params = len(param_names)
        
        for i, (param_name, value_scores) in enumerate(param_importance.items()):
            values = list(value_scores.keys())
            scores_for_param = list(value_scores.values())
            
            axes[0, 1].bar([f"{param_name}={v}" for v in values], scores_for_param, 
                          alpha=0.7, label=param_name)
        
        axes[0, 1].set_xlabel('Parameter Values')
        axes[0, 1].set_ylabel('Average MAE')
        axes[0, 1].set_title('Parameter Impact Analysis')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True)
    
    # 3. 分数排序
    sorted_indices = np.argsort(scores)
    sorted_scores = [scores[i] for i in sorted_indices]
    
    axes[1, 0].plot(range(len(sorted_scores)), sorted_scores, 'o-', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Parameter Combination Rank')
    axes[1, 0].set_ylabel('Average Validation MAE')
    axes[1, 0].set_title('Sorted Cross-Validation Scores')
    axes[1, 0].grid(True)
    
    # 4. 最佳参数组合详情
    best_idx = np.argmin(scores)
    best_params = params_list[best_idx]
    
    param_text = "Best Parameters:\n"
    for param_name, param_value in best_params.items():
        param_text += f"{param_name}: {param_value}\n"
    param_text += f"\nBest Score: {min(scores):.4f}"
    
    axes[1, 1].text(0.1, 0.5, param_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='center', fontfamily='monospace')
    axes[1, 1].set_title('Best Parameter Combination')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"网格搜索分析图表已保存至: {save_path}") 