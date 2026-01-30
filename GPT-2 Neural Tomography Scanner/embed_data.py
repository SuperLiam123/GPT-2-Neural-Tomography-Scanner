#!/usr/bin/env python3
"""
将GPT-2体积数据嵌入到HTML文件中
"""

import json
import re

def main():
    # 读取体积数据
    print("Loading volume data...")
    with open('volume_data.json', 'r') as f:
        volume_json = json.load(f)
    
    # 读取HTML模板
    print("Loading HTML template...")
    with open('neural_tomography.html', 'r') as f:
        html_content = f.read()
    
    # 创建嵌入的JavaScript代码
    volume_data_js = f"""
        // Embedded GPT-2 volume data
        const EMBEDDED_VOLUME_DATA = {json.dumps(volume_json)};
        
        // Decompress and load volume data
        async function loadEmbeddedVolumeData() {{
            const compressed = atob(EMBEDDED_VOLUME_DATA.volume.data);
            const compressedArray = new Uint8Array(compressed.length);
            for (let i = 0; i < compressed.length; i++) {{
                compressedArray[i] = compressed.charCodeAt(i);
            }}
            
            // Decompress using pako
            const decompressed = pako.inflate(compressedArray);
            
            // Convert to Float32Array with proper scaling
            const vMin = EMBEDDED_VOLUME_DATA.volume.min;
            const vMax = EMBEDDED_VOLUME_DATA.volume.max;
            const size = EMBEDDED_VOLUME_DATA.volume.shape[0];
            const totalVoxels = size * size * size;
            
            const volumeData = new Float32Array(totalVoxels);
            for (let i = 0; i < totalVoxels; i++) {{
                // Convert from uint8 back to original range
                volumeData[i] = (decompressed[i] / 255) * (vMax - vMin) + vMin;
            }}
            
            return {{ data: volumeData, size: size }};
        }}
    """
    
    # 修改init函数以使用嵌入的数据
    modified_init = """
        // Initialize application
        async function init() {
            updateLoadingProgress('Decompressing volume data...');
            
            try {
                // Try to load embedded data first
                if (typeof EMBEDDED_VOLUME_DATA !== 'undefined') {
                    const loaded = await loadEmbeddedVolumeData();
                    volumeData = loaded.data;
                    volumeSize = loaded.size;
                } else {
                    // Fallback to synthetic data
                    volumeData = generateSyntheticVolume();
                }
            } catch (e) {
                console.error('Error loading embedded data:', e);
                volumeData = generateSyntheticVolume();
            }
            
            updateLoadingProgress('Initializing 3D renderer...');
            await init3D();
            
            updateLoadingProgress('Rendering slices...');
            renderAllSlices();
            
            updateLoadingProgress('Drawing transfer function...');
            drawTransferFunction();
            
            // Setup event listeners
            setupEventListeners();
            
            // Hide loading overlay
            document.getElementById('loadingOverlay').style.display = 'none';
            
            // Start animation loop
            animate();
        }
    """
    
    # 在pako脚本之后插入数据
    insert_point = '<!-- Pako for decompression -->\n    <script src="https://cdn.jsdelivr.net/npm/pako@2.1.0/dist/pako.min.js"></script>'
    
    data_script = f'\n\n    <script>\n{volume_data_js}\n    </script>'
    
    html_content = html_content.replace(insert_point, insert_point + data_script)
    
    # 替换init函数
    # 使用正则表达式找到并替换init函数
    init_pattern = r'// Initialize application\s+async function init\(\) \{[^}]+\{[^}]+\}[^}]+\}'
    
    # 更简单的方法：直接替换整个init函数
    old_init_start = '// Initialize application\n        async function init() {'
    old_init_end = '// Start animation loop\n            animate();\n        }'
    
    # 找到init函数的位置并替换
    start_idx = html_content.find(old_init_start)
    if start_idx != -1:
        end_idx = html_content.find(old_init_end, start_idx) + len(old_init_end)
        html_content = html_content[:start_idx] + modified_init + html_content[end_idx:]
    
    # 更新滑块最大值以匹配实际体积大小
    html_content = html_content.replace('max="127"', 'max="127"')  # 保持128大小
    
    # 保存最终HTML
    output_file = 'neural_tomography_final.html'
    print(f"Saving final HTML to {output_file}...")
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    # 计算文件大小
    import os
    file_size = os.path.getsize(output_file)
    print(f"Final file size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    print("Done!")

if __name__ == '__main__':
    main()
