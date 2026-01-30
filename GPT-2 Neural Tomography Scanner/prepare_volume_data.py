#!/usr/bin/env python3
"""
GPT-2 Neural Tomography - Volume Data Preparation
将GPT-2模型参数转换为3D体积数据用于医学影像风格可视化
"""

import numpy as np
from transformers import GPT2Model
from scipy.ndimage import zoom, gaussian_filter
import json
import base64
import gzip

def load_gpt2_weights():
    """加载GPT-2模型权重"""
    print("Loading GPT-2 model...")
    model = GPT2Model.from_pretrained('gpt2')
    
    # 收集所有参数信息
    layer_info = []
    all_weights = []
    
    for name, param in model.named_parameters():
        weights = param.detach().numpy().flatten()
        layer_info.append({
            'name': name,
            'shape': list(param.shape),
            'size': len(weights),
            'start_idx': len(all_weights) if isinstance(all_weights, list) else sum(len(w) for w in all_weights)
        })
        all_weights.append(weights)
        print(f"  {name}: {param.shape}, {len(weights)} params")
    
    all_weights = np.concatenate(all_weights)
    print(f"\nTotal parameters: {len(all_weights):,}")
    print(f"Min: {all_weights.min():.4f}, Max: {all_weights.max():.4f}")
    print(f"Mean: {all_weights.mean():.4f}, Std: {all_weights.std():.4f}")
    
    return all_weights, layer_info

def organize_by_structure(all_weights, layer_info, volume_size=256):
    """
    按模型结构组织参数到3D体积中
    Z轴：模型深度（从词嵌入到输出层）
    X轴：输入维度
    Y轴：输出维度
    """
    print(f"\nOrganizing weights into {volume_size}x{volume_size}x{volume_size} volume...")
    
    # 创建空体积
    volume = np.zeros((volume_size, volume_size, volume_size), dtype=np.float32)
    
    # GPT-2 small结构：
    # - wte: 词嵌入 (50257, 768)
    # - wpe: 位置嵌入 (1024, 768)
    # - 12个transformer层，每层包含：
    #   - ln_1, attn (c_attn, c_proj), ln_2, mlp (c_fc, c_proj)
    # - ln_f: 最终层归一化
    
    z_slices_per_layer = volume_size // 14  # 12 transformer layers + embeddings + final
    
    current_z = 0
    weight_idx = 0
    
    for info in layer_info:
        name = info['name']
        shape = info['shape']
        size = info['size']
        
        # 确定Z范围
        if 'wte' in name or 'wpe' in name:
            z_start = 0
            z_end = z_slices_per_layer
        elif 'ln_f' in name:
            z_start = volume_size - z_slices_per_layer
            z_end = volume_size
        else:
            # 提取层号
            layer_num = None
            for part in name.split('.'):
                if part.isdigit():
                    layer_num = int(part)
                    break
            if layer_num is not None:
                z_start = z_slices_per_layer + layer_num * (volume_size - 2 * z_slices_per_layer) // 12
                z_end = z_start + (volume_size - 2 * z_slices_per_layer) // 12
            else:
                z_start = current_z
                z_end = min(current_z + z_slices_per_layer, volume_size)
        
        # 获取该层的权重
        weights = all_weights[weight_idx:weight_idx + size]
        weight_idx += size
        
        # 将权重填充到对应的Z切片中
        z_range = z_end - z_start
        if z_range > 0 and len(weights) > 0:
            # 将权重reshape成2D然后填充到3D
            target_size = z_range * volume_size * volume_size
            
            if len(weights) >= target_size:
                # 降采样
                indices = np.linspace(0, len(weights) - 1, target_size).astype(int)
                resampled = weights[indices]
            else:
                # 上采样（使用插值）
                resampled = np.interp(
                    np.linspace(0, len(weights) - 1, target_size),
                    np.arange(len(weights)),
                    weights
                )
            
            # 填充到体积中
            volume[z_start:z_end, :, :] = resampled.reshape(z_range, volume_size, volume_size)
    
    return volume

def create_volume_simple(all_weights, volume_size=256):
    """
    简单方法：将所有权重按顺序填充到立方体中
    """
    print(f"\nCreating {volume_size}x{volume_size}x{volume_size} volume (simple method)...")
    
    total_voxels = volume_size ** 3
    
    # 降采样或上采样到目标大小
    if len(all_weights) > total_voxels:
        # 使用滑动窗口取统计值
        window_size = len(all_weights) // total_voxels
        remainder = len(all_weights) % total_voxels
        
        # 使用更智能的降采样：取局部最大绝对值（保留显著特征）
        resampled = np.zeros(total_voxels, dtype=np.float32)
        for i in range(total_voxels):
            start = i * window_size + min(i, remainder)
            end = start + window_size + (1 if i < remainder else 0)
            window = all_weights[start:end]
            # 取绝对值最大的值（保留符号）
            max_idx = np.argmax(np.abs(window))
            resampled[i] = window[max_idx]
    else:
        # 上采样
        resampled = np.interp(
            np.linspace(0, len(all_weights) - 1, total_voxels),
            np.arange(len(all_weights)),
            all_weights
        )
    
    # Reshape成立方体
    volume = resampled.reshape(volume_size, volume_size, volume_size)
    
    return volume

def normalize_to_hu(volume):
    """
    将参数值归一化到模拟CT的HU值范围
    """
    print("\nNormalizing to HU-like values...")
    
    # Z-score归一化
    volume_normalized = (volume - volume.mean()) / volume.std()
    
    # 缩放到HU范围 (-1000 到 +2000)
    volume_hu = volume_normalized * 500
    
    # 裁剪极端值
    volume_hu = np.clip(volume_hu, -1000, 2000)
    
    print(f"HU range: {volume_hu.min():.1f} to {volume_hu.max():.1f}")
    
    return volume_hu

def apply_smoothing(volume, sigma=0.5):
    """
    应用轻微高斯平滑确保连续性
    """
    print(f"\nApplying Gaussian smoothing (sigma={sigma})...")
    return gaussian_filter(volume, sigma=sigma)

def compress_volume(volume, quantize_bits=8):
    """
    压缩体积数据用于嵌入HTML
    """
    print("\nCompressing volume data...")
    
    # 量化到8位以减小体积
    v_min, v_max = volume.min(), volume.max()
    
    if quantize_bits == 8:
        # 归一化到0-255
        volume_uint8 = ((volume - v_min) / (v_max - v_min) * 255).astype(np.uint8)
        raw_bytes = volume_uint8.tobytes()
    else:
        # 使用16位
        volume_uint16 = ((volume - v_min) / (v_max - v_min) * 65535).astype(np.uint16)
        raw_bytes = volume_uint16.tobytes()
    
    # Gzip压缩
    compressed = gzip.compress(raw_bytes, compresslevel=9)
    
    # Base64编码
    b64_data = base64.b64encode(compressed).decode('ascii')
    
    print(f"Original size: {len(raw_bytes):,} bytes")
    print(f"Compressed size: {len(compressed):,} bytes")
    print(f"Base64 size: {len(b64_data):,} chars")
    print(f"Compression ratio: {len(raw_bytes) / len(compressed):.2f}x")
    
    return {
        'data': b64_data,
        'shape': list(volume.shape),
        'min': float(v_min),
        'max': float(v_max),
        'bits': quantize_bits
    }

def main():
    # 1. 加载GPT-2权重
    all_weights, layer_info = load_gpt2_weights()
    
    # 2. 构建3D体积 - 使用128大小以减小文件体积
    volume_size = 128  # 减小到128以便嵌入HTML
    volume = create_volume_simple(all_weights, volume_size)
    
    # 3. 归一化到HU值
    volume_hu = normalize_to_hu(volume)
    
    # 4. 应用轻微平滑
    volume_smooth = apply_smoothing(volume_hu, sigma=0.3)
    
    # 5. 压缩数据
    compressed_data = compress_volume(volume_smooth, quantize_bits=8)
    
    # 6. 保存元数据
    metadata = {
        'model': 'GPT-2 Small',
        'total_params': len(all_weights),
        'volume_size': volume_size,
        'layers': layer_info[:5],  # 只保存前几层信息作为示例
        'stats': {
            'original_min': float(all_weights.min()),
            'original_max': float(all_weights.max()),
            'original_mean': float(all_weights.mean()),
            'original_std': float(all_weights.std()),
        }
    }
    
    # 7. 保存到JSON文件
    output = {
        'metadata': metadata,
        'volume': compressed_data
    }
    
    with open('volume_data.json', 'w') as f:
        json.dump(output, f)
    
    print(f"\nSaved volume data to volume_data.json")
    print(f"File size: {len(json.dumps(output)):,} bytes")
    
    return output

if __name__ == '__main__':
    main()
