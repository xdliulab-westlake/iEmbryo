"""
PyTorch原生批量缓存系统
使用torch.save和分块存储，替代大量小文件的渐进式缓存
"""

import os
import json
import hashlib
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import time
# 预处理函数在需要时动态导入


class NumpyEncoder(json.JSONEncoder):
    """处理numpy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class BatchCacheManager:
    """
    批量缓存管理器
    将图片分块存储为PyTorch tensor文件，大幅减少文件数量
    """

    def __init__(self, cache_dir: str = "./batch_cache", chunk_size: int = 1000):
        """
        初始化批量缓存管理器

        Args:
            cache_dir: 缓存目录
            chunk_size: 每个文件存储的图片数量
        """
        self.cache_dir = Path(cache_dir)
        self.chunk_size = chunk_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 元数据文件
        self.metadata_file = self.cache_dir / "metadata.json"
        self.index_file = self.cache_dir / "index.json"

        # 加载现有元数据
        self.metadata = self._load_metadata()
        self.index = self._load_index()

    def _load_metadata(self) -> Dict:
        """加载缓存元数据"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            "version": "1.0",
            "chunk_size": self.chunk_size,
            "total_chunks": 0,
            "total_images": 0,
            "img_size": None,
            "created_time": time.time()
        }

    def _load_index(self) -> Dict:
        """加载图片路径索引"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """保存元数据"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, cls=NumpyEncoder)

    def _save_index(self):
        """保存索引"""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2, cls=NumpyEncoder)

    def _get_cache_key(self, data_dir: str, img_size: int,
                      apply_petri_crop: bool, apply_center_crop: bool,
                      split: str = "train", train_split: float = 0.7,
                      random_state: int = 42) -> str:
        """
        生成稳定的缓存键
        基于数据目录和配置参数而不是具体路径列表，确保跨运行一致性
        """
        # 包含所有影响预处理结果的参数
        params = [
            f"data_dir={data_dir}",
            f"img_size={img_size}",
            f"petri_crop={apply_petri_crop}",
            f"center_crop={apply_center_crop}",
            f"split={split}",
            f"train_split={train_split}",
            f"random_state={random_state}",
            f"version=v2.0"  # 版本号，升级缓存格式时可修改
        ]

        params_str = "|".join(params)
        cache_key = hashlib.md5(params_str.encode()).hexdigest()[:16]
        return cache_key

    def _get_chunk_filename(self, chunk_id: int, split: str = "train") -> str:
        """获取分块文件名"""
        return f"{split}_chunk_{chunk_id:04d}.pt"

    def is_cached(self, data_dir: str, img_size: int,
                  apply_petri_crop: bool = True, apply_center_crop: bool = True,
                  split: str = "train", train_split: float = 0.7,
                  random_state: int = 42) -> bool:
        """检查是否已缓存"""
        cache_key = self._get_cache_key(data_dir, img_size, apply_petri_crop,
                                      apply_center_crop, split, train_split, random_state)
        return cache_key in self.index

    def save_batch_cache(self, image_paths: List[str], labels: List[int],
                        data_dir: str, img_size: int,
                        apply_petri_crop: bool = True, apply_center_crop: bool = True,
                        split: str = "train", is_training: bool = True,
                        train_split: float = 0.7, random_state: int = 42) -> str:
        """
        保存批量缓存

        Args:
            image_paths: 图片路径列表
            labels: 标签列表
            img_size: 图片尺寸
            apply_petri_crop: 是否应用培养皿裁剪
            apply_center_crop: 是否应用中心裁剪
            split: 数据分割类型 (train/test)
            is_training: 是否为训练集

        Returns:
            缓存键
        """
        cache_key = self._get_cache_key(data_dir, img_size, apply_petri_crop,
                                      apply_center_crop, split, train_split, random_state)

        if cache_key in self.index:
            print(f"缓存已存在: {cache_key}")
            return cache_key

        print(f"开始批量缓存: {len(image_paths)} 张图片")
        start_time = time.time()

        # 分块处理
        chunks_info = []
        total_chunks = (len(image_paths) + self.chunk_size - 1) // self.chunk_size

        for chunk_id in range(total_chunks):
            start_idx = chunk_id * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(image_paths))

            chunk_paths = image_paths[start_idx:end_idx]
            chunk_labels = labels[start_idx:end_idx]

            # 预处理这一批图片
            chunk_data = []
            for path in chunk_paths:
                try:
                    # 使用现有的预处理函数
                    from .stork_preprocessing import preprocess_embryo_image
                    import numpy as np

                    # 调用现有的预处理函数
                    processed_img = preprocess_embryo_image(
                        path, img_size, apply_petri_crop, apply_center_crop
                    )

                    # 优化存储：保持uint8格式，节省50%存储空间
                    # 训练时再转换为float32并归一化
                    preprocessed_img = np.transpose(processed_img, (2, 0, 1))  # HWC -> CHW

                    chunk_data.append(torch.from_numpy(preprocessed_img.astype(np.uint8)))
                except Exception as e:
                    print(f"预处理失败: {path}, 错误: {e}")
                    # 创建零图像作为占位符（uint8格式）
                    chunk_data.append(torch.zeros(3, img_size, img_size, dtype=torch.uint8))

            # 保存这一块
            chunk_filename = self._get_chunk_filename(chunk_id, split)
            chunk_path = self.cache_dir / chunk_filename

            chunk_dict = {
                'images': torch.stack(chunk_data),  # [N, 3, H, W]
                'labels': torch.tensor(chunk_labels),  # [N]
                'paths': chunk_paths,
                'chunk_id': chunk_id,
                'start_idx': start_idx,
                'end_idx': end_idx
            }

            # 使用压缩保存，进一步减少存储空间
            torch.save(chunk_dict, chunk_path, _use_new_zipfile_serialization=False)

            chunks_info.append({
                'chunk_id': chunk_id,
                'filename': chunk_filename,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'size': end_idx - start_idx
            })

            print(f"   保存块 {chunk_id+1}/{total_chunks}: {chunk_filename} ({end_idx-start_idx} 张)")

        # 更新索引
        self.index[cache_key] = {
            'image_paths': list(image_paths) if hasattr(image_paths, '__iter__') else image_paths,  # 确保转换为list
            'labels': [int(label) for label in labels],  # 确保转换为Python int
            'img_size': int(img_size),
            'apply_petri_crop': bool(apply_petri_crop),
            'apply_center_crop': bool(apply_center_crop),
            'split': split,
            'chunks': chunks_info,
            'total_images': len(image_paths),
            'created_time': float(time.time())
        }

        # 更新元数据
        self.metadata['total_chunks'] += total_chunks
        self.metadata['total_images'] += len(image_paths)
        self.metadata['img_size'] = int(img_size)

        # 保存索引和元数据
        self._save_index()
        self._save_metadata()

        elapsed_time = time.time() - start_time
        print(f"批量缓存完成: {cache_key}")
        print(f"   文件数量: {total_chunks} 个chunk文件 (vs {len(image_paths)} 个单独文件)")
        print(f"   缓存时间: {elapsed_time:.2f} 秒")
        print(f"   平均速度: {len(image_paths)/elapsed_time:.1f} 张/秒")

        return cache_key

    def load_cached_data(self, cache_key: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, List[str]]]:
        """
        加载缓存数据

        Args:
            cache_key: 缓存键

        Returns:
            (images, labels, paths) 或 None
        """
        if cache_key not in self.index:
            return None

        cache_info = self.index[cache_key]
        chunks = cache_info['chunks']

        print(f"加载批量缓存: {cache_key}")
        start_time = time.time()

        all_images = []
        all_labels = []
        all_paths = []

        for chunk_info in chunks:
            chunk_path = self.cache_dir / chunk_info['filename']

            if not chunk_path.exists():
                print(f"缓存块不存在: {chunk_info['filename']}")
                return None

            try:
                chunk_data = torch.load(chunk_path, map_location='cpu')
                all_images.append(chunk_data['images'])
                all_labels.append(chunk_data['labels'])
                all_paths.extend(chunk_data['paths'])
            except Exception as e:
                print(f"加载缓存块失败: {chunk_info['filename']}, 错误: {e}")
                return None

        # 检查是否有数据
        if not all_images:
            print("缓存中没有图片数据")
            return None

        # 合并所有数据
        images = torch.cat(all_images, dim=0)
        labels = torch.cat(all_labels, dim=0)

        elapsed_time = time.time() - start_time
        print(f"批量缓存加载完成: {len(images)} 张图片")
        print(f"   加载时间: {elapsed_time:.2f} 秒")
        print(f"   加载速度: {len(images)/elapsed_time:.1f} 张/秒")

        return images, labels, all_paths

    def clear_cache(self, cache_key: Optional[str] = None):
        """清理缓存"""
        if cache_key is None:
            # 清理所有缓存
            for file in self.cache_dir.glob("*.pt"):
                file.unlink()
            self.index.clear()
            self.metadata['total_chunks'] = 0
            self.metadata['total_images'] = 0
        else:
            # 清理特定缓存
            if cache_key in self.index:
                cache_info = self.index[cache_key]
                for chunk_info in cache_info['chunks']:
                    chunk_path = self.cache_dir / chunk_info['filename']
                    if chunk_path.exists():
                        chunk_path.unlink()
                del self.index[cache_key]

        self._save_index()
        self._save_metadata()
        print(f"缓存清理完成")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_size = 0
        for file in self.cache_dir.glob("*.pt"):
            total_size += file.stat().st_size

        return {
            'total_caches': len(self.index),
            'total_chunks': self.metadata.get('total_chunks', 0),
            'total_images': self.metadata.get('total_images', 0),
            'total_size_mb': total_size / (1024 * 1024),
            'avg_size_per_image_kb': (total_size / max(1, self.metadata.get('total_images', 1))) / 1024,
            'cache_dir': str(self.cache_dir)
        }


class BatchCachedDataset(torch.utils.data.Dataset):
    """
    基于批量缓存的数据集
    自动使用批量缓存或按需创建
    支持内存映射和延迟加载以提升性能
    """

    def __init__(self, image_paths: List[str], labels: List[int], img_size: int = 224,
                 is_training: bool = True, apply_petri_crop: bool = True,
                 apply_center_crop: bool = True, cache_dir: str = "./batch_cache",
                 chunk_size: int = 1000, split: str = "train",
                 use_memory_mapping: bool = False, data_dir: str = None,
                 train_split: float = 0.7, random_state: int = 42):
        """
        初始化批量缓存数据集

        Args:
            image_paths: 图片路径列表
            labels: 标签列表
            img_size: 图片尺寸
            is_training: 是否为训练集
            apply_petri_crop: 是否应用培养皿裁剪
            apply_center_crop: 是否应用中心裁剪
            cache_dir: 缓存目录
            chunk_size: 每个缓存文件的图片数量
            split: 数据分割名称
            use_memory_mapping: 是否使用内存映射（大数据集推荐）
        """
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        self.is_training = is_training
        self.apply_petri_crop = apply_petri_crop
        self.apply_center_crop = apply_center_crop
        self.split = split
        self.use_memory_mapping = use_memory_mapping
        self.data_dir = data_dir
        self.train_split = train_split
        self.random_state = random_state

        # 初始化缓存管理器
        self.cache_manager = BatchCacheManager(cache_dir, chunk_size)

        # 生成稳定的缓存键
        self.cache_key = self.cache_manager._get_cache_key(
            data_dir, img_size, apply_petri_crop, apply_center_crop,
            split, train_split, random_state
        )

        # 检查是否有图片
        if len(image_paths) == 0:
            print(f"{split}数据集为空，创建空数据集")
            self.images = torch.empty(0, 3, img_size, img_size)
            self.cached_labels = torch.empty(0, dtype=torch.long)
            self.cached_paths = []
            return

        self.cached_data = None
        if self.cache_manager.is_cached(data_dir, img_size, apply_petri_crop, apply_center_crop,
                                      split, train_split, random_state):
            print(f"发现已有缓存: {self.cache_key}，加载中...")
            self.cached_data = self.cache_manager.load_cached_data(self.cache_key)
        else:
            print(f"未发现缓存: {self.cache_key}，创建批量缓存...")
            self.cache_manager.save_batch_cache(
                image_paths, labels, data_dir, img_size, apply_petri_crop,
                apply_center_crop, split, is_training, train_split, random_state
            )
            self.cached_data = self.cache_manager.load_cached_data(self.cache_key)

        if self.cached_data is None:
            print(f"无法加载缓存数据，创建空数据集")
            self.images = torch.empty(0, 3, img_size, img_size, dtype=torch.uint8)
            self.cached_labels = torch.empty(0, dtype=torch.long)
            self.cached_paths = []
        else:
            if self.use_memory_mapping:
                # 延迟加载：只保存元数据，使用时再加载具体chunk
                print(f" 启用延迟加载模式")
                self.images = None  # 延迟加载
                self.cached_labels = self.cached_data[1]
                self.cached_paths = self.cached_data[2]
                self._setup_chunk_mapping()
            else:
                # 全量加载到内存
                self.images, self.cached_labels, self.cached_paths = self.cached_data
                print(f"批量缓存数据集初始化完成: {len(self.images)} 张图片")

    def _setup_chunk_mapping(self):
        """设置chunk映射，用于延迟加载"""
        cache_info = self.cache_manager.index[self.cache_key]
        self.chunk_mapping = {}  # idx -> (chunk_id, chunk_offset)

        current_idx = 0
        for chunk_info in cache_info['chunks']:
            chunk_size = chunk_info['size']
            for offset in range(chunk_size):
                self.chunk_mapping[current_idx] = (chunk_info['chunk_id'], offset)
                current_idx += 1

        self.chunk_cache = {}  # 缓存已加载的chunk

    def _load_chunk(self, chunk_id: int) -> torch.Tensor:
        """加载指定的chunk"""
        if chunk_id in self.chunk_cache:
            return self.chunk_cache[chunk_id]

        cache_info = self.cache_manager.index[self.cache_key]
        chunk_info = next(c for c in cache_info['chunks'] if c['chunk_id'] == chunk_id)
        chunk_path = self.cache_manager.cache_dir / chunk_info['filename']

        chunk_data = torch.load(chunk_path, map_location='cpu')
        self.chunk_cache[chunk_id] = chunk_data['images']

        # 限制缓存大小，避免内存溢出
        if len(self.chunk_cache) > 3:  # 最多缓存3个chunk
            oldest_chunk = min(self.chunk_cache.keys())
            del self.chunk_cache[oldest_chunk]

        return chunk_data['images']

    def __len__(self):
        if self.use_memory_mapping:
            return len(self.cached_labels)
        else:
            return len(self.images)

    def __getitem__(self, idx):
        """获取数据项"""
        if self.use_memory_mapping:
            # 延迟加载模式
            chunk_id, chunk_offset = self.chunk_mapping[idx]
            chunk_images = self._load_chunk(chunk_id)
            image = chunk_images[chunk_offset]
        else:
            # 全量加载模式
            image = self.images[idx]  # uint8格式的预处理tensor

        label = self.cached_labels[idx]

        # 转换为float32并归一化到[0,1]
        image = image.float() / 255.0

        return image, label.long()

    def get_cache_stats(self):
        """获取缓存统计信息"""
        return self.cache_manager.get_cache_stats()