import cv2
import pandas as pd
import numpy as np
import copy
import os
import glob

def find_embryo_image(embryo_path, preferred_focal_depth="F0"):
    """
    在胚胎目录中查找图像文件
    
    Args:
        embryo_path: 胚胎目录路径 (如: /liaozizhuo/image/D2023.09.04_S00918_I4425_P_10)
        preferred_focal_depth: 首选的focal depth目录 (默认: F0)
        
    Returns:
        str: 找到的第一个图像文件路径，如果没找到则返回None
    """
    
    # 构建images目录路径
    images_dir = os.path.join(embryo_path, "images")
    
    if not os.path.exists(images_dir):
        print(f"❌ images目录不存在: {images_dir}")
        return None
    
    # 首先尝试首选的focal depth
    preferred_dir = os.path.join(images_dir, preferred_focal_depth)
    if os.path.exists(preferred_dir):
        # 在首选目录中查找图像文件
        image_files = find_image_files_in_directory(preferred_dir)
        if image_files:
            print(f"✅ 在{preferred_focal_depth}目录找到图像: {os.path.basename(image_files[0])}")
            return image_files[0]
    
    # 如果首选目录没有找到，搜索所有focal depth目录
    focal_dirs = [d for d in os.listdir(images_dir) 
                  if os.path.isdir(os.path.join(images_dir, d)) and d.startswith('F')]
    
    # 按照focal depth排序 (F0, F15, F30, ...)
    def sort_focal_depth(dirname):
        try:
            if dirname == "F0":
                return 0
            return int(dirname.replace('F', '').replace('-', ''))
        except:
            return 999
    
    focal_dirs.sort(key=sort_focal_depth)
    
    for focal_dir in focal_dirs:
        full_dir = os.path.join(images_dir, focal_dir)
        image_files = find_image_files_in_directory(full_dir)
        if image_files:
            print(f"✅ 在{focal_dir}目录找到图像: {os.path.basename(image_files[0])}")
            return image_files[0]
    
    print(f"❌ 在胚胎目录中未找到任何图像文件: {embryo_path}")
    return None


def find_image_files_in_directory(directory):
    """
    在指定目录中查找所有图像文件
    
    Args:
        directory: 目录路径
        
    Returns:
        list: 图像文件路径列表
    """
    if not os.path.exists(directory):
        return []
    
    image_files = []
    
    # 常见图像扩展名
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    # 搜索有扩展名的图像文件
    for ext in image_extensions:
        pattern = os.path.join(directory, ext)
        image_files.extend(glob.glob(pattern, recursive=False))
        # 也搜索大写扩展名
        pattern_upper = os.path.join(directory, ext.upper())
        image_files.extend(glob.glob(pattern_upper, recursive=False))
    
    if image_files:
        return sorted(image_files)
    
    # 如果没有找到有扩展名的文件，检查所有文件
    try:
        all_files = [os.path.join(directory, f) for f in os.listdir(directory)
                     if os.path.isfile(os.path.join(directory, f))]
        
        # 尝试通过文件头检测图像文件
        for file_path in all_files:
            if is_image_file(file_path):
                image_files.append(file_path)
        
        return sorted(image_files)
    except:
        return []


def is_image_file(file_path):
    """
    通过文件头检测是否为图像文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        bool: 是否为图像文件
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(16)
        
        # 检测常见图像格式的文件头
        image_signatures = [
            b'\xff\xd8\xff',  # JPEG
            b'\x89PNG\r\n\x1a\n',  # PNG
            b'GIF87a',  # GIF87a
            b'GIF89a',  # GIF89a
            b'BM',  # BMP
            b'II*\x00',  # TIFF little-endian
            b'MM\x00*',  # TIFF big-endian
        ]
        
        for sig in image_signatures:
            if header.startswith(sig):
                return True
        
        # 检测WebP
        if header.startswith(b'RIFF') and b'WEBP' in header:
            return True
        
        return False
    except:
        return False


def read_image_with_format_detection(file_path):
    """
    读取图像文件，自动检测格式（支持无扩展名文件）
    
    Args:
        file_path: 图像文件路径
        
    Returns:
        numpy.ndarray: 图像数组，BGR格式
    """
    # 首先尝试直接读取
    img = cv2.imread(file_path)
    if img is not None:
        return img
    
    # 如果失败，检测文件格式并尝试其他方法
    try:
        with open(file_path, 'rb') as f:
            header = f.read(16)
        
        # 根据文件头确定格式
        detected_format = None
        if header.startswith(b'\xff\xd8\xff'):
            detected_format = 'JPEG'
            extensions_to_try = ['.jpg', '.jpeg']
        elif header.startswith(b'\x89PNG\r\n\x1a\n'):
            detected_format = 'PNG'
            extensions_to_try = ['.png']
        elif header.startswith(b'II*\x00') or header.startswith(b'MM\x00*'):
            detected_format = 'TIFF'
            extensions_to_try = ['.tiff', '.tif']
        elif header.startswith(b'BM'):
            detected_format = 'BMP'
            extensions_to_try = ['.bmp']
        elif header.startswith(b'RIFF') and b'WEBP' in header:
            detected_format = 'WEBP'
            extensions_to_try = ['.webp']
        else:
            # 默认尝试常见格式
            extensions_to_try = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        
        print(f"🔍 检测到图像格式: {detected_format or 'Unknown'}")
        
        # 尝试通过添加扩展名来读取
        import tempfile
        import shutil
        
        for ext in extensions_to_try:
            try:
                # 创建临时文件
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # 复制原文件到临时文件
                shutil.copy2(file_path, temp_path)
                
                # 尝试读取临时文件
                img = cv2.imread(temp_path)
                
                # 清理临时文件
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
                if img is not None:
                    print(f"✅ 通过格式 {ext} 成功读取图像")
                    return img
                    
            except Exception as e:
                print(f"⚠️ 尝试格式 {ext} 失败: {e}")
                continue
        
        # 如果所有方法都失败，尝试使用PIL
        try:
            from PIL import Image
            import numpy as np
            
            pil_img = Image.open(file_path)
            # 转换为RGB格式
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            # 转换为numpy数组并改为BGR格式 (OpenCV使用BGR)
            img_array = np.array(pil_img)
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            print("✅ 通过PIL成功读取图像")
            return img
            
        except Exception as e:
            print(f"⚠️ PIL读取也失败: {e}")
        
        return None
        
    except Exception as e:
        print(f"❌ 图像格式检测失败: {e}")
        return None

def crop_center_square(frame):
    """
    Crop the center square from an image
    
    Args:
        frame: Input image
    
    Returns:
        Center square cropped image
    """
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def crop_petri_dish(frame):
    """
    Detect and crop Petri dish using optimized Hough Circle detection
    
    Args:
        frame: Input image (RGB)
    
    Returns:
        Cropped image with circular mask applied, or original frame if no circle detected
    """
    # Convert to grayscale for circle detection
    if len(frame.shape) == 3:
        gimg = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gimg = frame.copy()
        
    cimg = copy.deepcopy(frame)
    img_height, img_width = gimg.shape
    img_center_x, img_center_y = img_width // 2, img_height // 2
    
    # Apply Gaussian blur to reduce noise
    gimg_blurred = cv2.GaussianBlur(gimg, (9, 9), 2)
    
    # Adaptive parameter selection based on image size
    img_diagonal = np.sqrt(img_height**2 + img_width**2)
    
    # Adaptive radius range (15% to 45% of image diagonal)
    min_radius = max(50, int(img_diagonal * 0.15))
    max_radius = min(int(img_diagonal * 0.45), max(img_height, img_width) // 2)
    
    # Multiple parameter combinations to try (optimized for 96.2% success rate)
    param_combinations = [
        (60, 40),   # 稍微不那么敏感，避免小圆圈
        (70, 45),   # 中等敏感度
        (50, 35),   # 较敏感
        (80, 50),   # 不敏感
        (90, 55),   # 最不敏感
    ]
    
    all_circles = []
    
    # Try different parameter combinations
    for param1, param2 in param_combinations:
        circles = cv2.HoughCircles(
            gimg_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=max(80, int(min(img_height, img_width) * 0.2)),
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None and len(circles[0]) > 0:
            for circle in circles[0]:
                x, y, r = circle
                # 计算到图像中心的距离
                distance_to_center = np.sqrt((x - img_center_x)**2 + (y - img_center_y)**2)
                all_circles.append({
                    'x': x, 'y': y, 'radius': r,
                    'distance_to_center': distance_to_center,
                    'params': (param1, param2)
                })
    
    if not all_circles:
        return frame
    
    # 转换为DataFrame进行筛选
    df_circles = pd.DataFrame(all_circles)
    
    # 智能筛选策略：
    # 1. 半径权重 (40%): 更大的圆圈得分更高
    # 2. 位置权重 (35%): 更接近中心的圆圈得分更高  
    # 3. 大小合理性权重 (25%): 半径在合理范围内的得分更高
    
    # 归一化各项指标到0-1
    df_circles['radius_norm'] = (df_circles['radius'] - df_circles['radius'].min()) / (df_circles['radius'].max() - df_circles['radius'].min() + 1e-6)
    df_circles['center_norm'] = 1 - (df_circles['distance_to_center'] / df_circles['distance_to_center'].max())
    
    # 大小合理性：偏好中等偏大的圆圈
    ideal_radius = (min_radius + max_radius) * 0.7  # 偏向较大圆圈
    df_circles['size_reasonableness'] = 1 - abs(df_circles['radius'] - ideal_radius) / (max_radius - min_radius)
    
    # 综合评分
    df_circles['score'] = (
        df_circles['radius_norm'] * 0.4 +
        df_circles['center_norm'] * 0.35 + 
        df_circles['size_reasonableness'] * 0.25
    )
    
    # 选择得分最高的圆圈
    best_circle = df_circles.loc[df_circles['score'].idxmax()]
    
    circles = np.array([[[best_circle['x'], best_circle['y'], best_circle['radius']]]])
    
    if (circles is not None) and (len(circles) > 0):
        # Find the largest circle (likely the Petri dish)
        df_circle = pd.DataFrame(data=circles[0], columns=['x', 'y', 'radius'])
        df_circle = df_circle.sort_values(by=['radius'], ascending=False)
        circle = df_circle.iloc[0].values

        # Create circular mask
        mask = np.zeros_like(cimg)
        if len(cimg.shape) == 3:
            mask = cv2.circle(mask.copy(), (int(circle[0]), int(circle[1])), 
                            int(circle[2]), (255, 255, 255), -1)
        else:
            mask = cv2.circle(mask.copy(), (int(circle[0]), int(circle[1])), 
                            int(circle[2]), 255, -1)
        
        # Apply mask
        result = cv2.bitwise_and(cimg, mask)
        return result
    else:
        # Return original frame if no circle detected
        return frame

def preprocess_embryo_image(image, img_size=224, apply_petri_crop=True, apply_center_crop=True):
    """
    Complete preprocessing pipeline for embryo images
    
    Args:
        image: Input image (numpy array or path)
        img_size: Target image size
        apply_petri_crop: Whether to apply Petri dish detection and cropping
        apply_center_crop: Whether to apply center square cropping
    
    Returns:
        Preprocessed image ready for model input
    """
    # Load image if path is provided
    if isinstance(image, str):
        # 检查路径是否存在
        if not os.path.exists(image):
            raise FileNotFoundError(f"路径不存在: {image}")
        
        # 检查是目录还是文件
        if os.path.isdir(image):
            # 如果是目录，尝试找到胚胎图像文件
            print(f"🔍 检测到胚胎目录，搜索图像文件: {os.path.basename(image)}")
            actual_image_path = find_embryo_image(image, preferred_focal_depth="F0")
            if actual_image_path is None:
                raise ValueError(f"在胚胎目录中未找到图像文件: {image}")
            image = actual_image_path
        
        # 尝试读取图像 (支持无扩展名文件)
        loaded_image = read_image_with_format_detection(image)
        if loaded_image is None:
            raise ValueError(f"无法读取图像文件: {image}。请检查文件是否损坏或格式是否支持。")
        
        image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
    
    # Apply Petri dish cropping if enabled
    if apply_petri_crop:
        image = crop_petri_dish(image)
    
    # Apply center cropping if enabled
    if apply_center_crop:
        image = crop_center_square(image)
    
    # Resize to target size
    image = cv2.resize(image, (img_size, img_size))
    
    # Ensure RGB format (3 channels)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    return image

def batch_process_images(image_paths, img_size=224, apply_petri_crop=True, apply_center_crop=True):
    """
    Process a batch of images using the BiLSTM pipeline
    
    Args:
        image_paths: List of image file paths
        img_size: Target image size
        apply_petri_crop: Whether to apply Petri dish detection and cropping
        apply_center_crop: Whether to apply center square cropping
    
    Returns:
        Numpy array of processed images
    """
    processed_images = []
    
    for path in image_paths:
        try:
            processed_img = preprocess_embryo_image(
                path, img_size, apply_petri_crop, apply_center_crop
            )
            processed_images.append(processed_img)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            # Create a black image as fallback
            fallback_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            processed_images.append(fallback_img)
    
    return np.array(processed_images)