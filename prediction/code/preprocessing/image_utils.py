# copy_images_by_split.py
import tensorflow as tf
import os
import re
import shutil

def copy_images_by_split(image_paths, X_train, y_train, X_val, y_val, X_test, y_test, 
                         train_embryos, val_embryos, test_embryos,
                         output_root='visualized_split'):

    def extract_embryo_id(path):
        if isinstance(path, tf.Tensor):
            path = path.numpy().decode("utf-8") 
        filename = os.path.basename(path)
        match = re.search(r'(.*?)_P', filename)
        return match.group(1) if match else None

    embryo_to_images = {}
    for path in image_paths:
        emb_id = extract_embryo_id(path)
        if emb_id:
            embryo_to_images.setdefault(emb_id, []).append(path)

    split_map = {
        'train': train_embryos,
        'val': val_embryos,
        'test': test_embryos
    }

    print(f"Creating output directory structure in {output_root}")
    for split in split_map.keys():
        for label in ['class_0', 'class_1']:
            os.makedirs(os.path.join(output_root, split, label), exist_ok=True)

    print("Copying images...")
    for split, embryo_list in split_map.items():
        for embryo_id in embryo_list:
            for path in embryo_to_images.get(embryo_id, []):
                if isinstance(path, tf.Tensor):
                    path = path.numpy().decode("utf-8")
                label = 'class_1' if 'class_1' in path else 'class_0'
                dst_dir = os.path.join(output_root, split, label)
                dst_path = os.path.join(dst_dir, os.path.basename(path))
                try:
                    shutil.copy(path, dst_path)
                except Exception as e:
                    print(f"Failed to copy {path} to {dst_path}: {e}")
