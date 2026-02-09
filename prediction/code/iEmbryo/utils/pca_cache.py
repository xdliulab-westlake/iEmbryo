#!/usr/bin/env python3
"""
PCA缓存管理器 - 确保多个脚本使用相同的数据和PCA变换
"""

import pickle
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA


class PCACache:
    """管理PCA模型和数据的缓存"""

    def __init__(self, cache_dir='outputs/pca_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_config_hash(self, config_dict):
        """根据配置生成唯一哈希值"""
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def save(self, features, labels, paths, pca_model, config_dict):
        """
        保存特征、PCA模型和元数据

        Args:
            features: 原始特征 (N, D)
            labels: 标签 (N,)
            paths: 图片路径列表
            pca_model: 训练好的PCA模型
            config_dict: 配置字典（用于生成唯一ID）
        """
        cache_id = self._get_config_hash(config_dict)
        cache_file = self.cache_dir / f'pca_cache_{cache_id}.pkl'

        cache_data = {
            'features': features,
            'labels': labels,
            'paths': paths,
            'pca_model': pca_model,
            'config': config_dict,
            'features_pca': pca_model.transform(features),
            'explained_variance_ratio': pca_model.explained_variance_ratio_,
            'n_components': pca_model.n_components_
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"\n✅ PCA缓存已保存: {cache_file}")
        print(f"   缓存ID: {cache_id}")
        print(f"   样本数: {len(features)}")
        print(f"   特征维度: {features.shape[1]}")
        print(f"   PCA维度: {pca_model.n_components_}")
        print(f"   方差解释率: {pca_model.explained_variance_ratio_}")

        return cache_id

    def load(self, config_dict=None, cache_id=None):
        """
        加载缓存的PCA数据

        Args:
            config_dict: 配置字典（自动查找对应缓存）
            cache_id: 直接指定缓存ID

        Returns:
            cache_data: 包含features, labels, paths, pca_model等的字典
        """
        if cache_id is None and config_dict is not None:
            cache_id = self._get_config_hash(config_dict)
        elif cache_id is None:
            raise ValueError("必须提供 config_dict 或 cache_id")

        cache_file = self.cache_dir / f'pca_cache_{cache_id}.pkl'

        if not cache_file.exists():
            return None

        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        print(f"\n✅ 已加载PCA缓存: {cache_file}")
        print(f"   缓存ID: {cache_id}")
        print(f"   样本数: {len(cache_data['features'])}")
        print(f"   PCA维度: {cache_data['n_components']}")

        return cache_data

    def list_caches(self):
        """列出所有可用的缓存"""
        cache_files = list(self.cache_dir.glob('pca_cache_*.pkl'))

        if not cache_files:
            print("没有找到缓存文件")
            return []

        print(f"\n可用的PCA缓存 ({len(cache_files)}个):")
        print("="*80)

        cache_infos = []
        for cache_file in cache_files:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            cache_id = cache_file.stem.replace('pca_cache_', '')
            info = {
                'cache_id': cache_id,
                'n_samples': len(cache_data['features']),
                'n_components': cache_data['n_components'],
                'config': cache_data['config']
            }
            cache_infos.append(info)

            print(f"ID: {cache_id}")
            print(f"  样本数: {info['n_samples']}")
            print(f"  PCA维度: {info['n_components']}")
            print(f"  配置: {json.dumps(info['config'], indent=4)}")
            print("-"*80)

        return cache_infos


def create_pca_config(model_path, csv_path, base_dir, fair_dir,
                      poor_csv_path, poor_base_dir,
                      max_samples_per_class, max_fair_samples,
                      random_seed=42):
    """
    创建标准化的PCA配置字典

    Args:
        model_path: 模型路径
        csv_path: Good胚胎CSV路径
        base_dir: Good胚胎基础目录
        fair_dir: Fair胚胎目录
        poor_csv_path: Poor胚胎CSV路径
        poor_base_dir: Poor胚胎基础目录
        max_samples_per_class: 每类最大样本数
        max_fair_samples: Fair类最大样本数
        random_seed: 随机种子

    Returns:
        config_dict: 配置字典
    """
    return {
        'model_path': str(Path(model_path).resolve()),
        'csv_path': str(Path(csv_path).resolve()) if csv_path else None,
        'base_dir': str(Path(base_dir).resolve()) if base_dir else None,
        'fair_dir': str(Path(fair_dir).resolve()) if fair_dir else None,
        'poor_csv_path': str(Path(poor_csv_path).resolve()) if poor_csv_path else None,
        'poor_base_dir': str(Path(poor_base_dir).resolve()) if poor_base_dir else None,
        'max_samples_per_class': max_samples_per_class,
        'max_fair_samples': max_fair_samples,
        'random_seed': random_seed
    }


if __name__ == '__main__':
    # 测试代码
    cache = PCACache()
    cache.list_caches()
