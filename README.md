# GPT-2 Neural Tomography Scanner

**"Scanning the mind of AI"**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project provides an interactive visualization tool that simulates a CT/MRI scan of a GPT-2 Small model's parameters. It treats the 124 million parameters of the neural network as a "digital life form" and allows you to explore its internal structure through tomographic slices and 3D volume rendering, all within a single, self-contained HTML file.

## Features

- **Interactive 3D Volume Rendering**: Explore the full parameter space of GPT-2 as a point cloud.
- **Orthogonal Slice Views**: Inspect the model's "anatomy" through three synchronized views (Axial, Coronal, Sagittal).
- **Medical Imaging Controls**: 
    - **Windowing**: Adjust window center (level) and width to highlight different parameter ranges.
    - **Presets**: Quickly switch between window settings like "Soft Tissue" and "High Density".
    - **Transfer Functions**: Apply different color maps (e.g., MRI T1, CT Bone, Heatmap) to enhance structural details.
- **Real-time Interaction**: All controls update the visualization instantly, running smoothly in your browser.
- **Single-File Application**: The entire tool is packed into a single HTML file. No installation or dependencies required.

## How It Works

The core idea is to map the one-dimensional array of GPT-2's 124 million parameters into a 3D cube (128x128x128 voxels). This 3D volume is then rendered using techniques inspired by medical imaging software.

1.  **Data Synthesis**: To keep the final file size manageable, the application uses a sophisticated algorithm to generate synthetic data that mimics the structural properties of GPT-2's parameters. This includes distinct patterns for embedding layers, attention mechanisms, and MLP blocks across the 12 transformer layers.
2.  **Volume Rendering**: The 3D view is created using Three.js. It renders the volume as a point cloud, with color and opacity determined by the selected transfer function.
3.  **Slice Generation**: The 2D slice views are generated on-the-fly by sampling the 3D volume data at the selected slice position for each axis.
4.  **Self-Contained HTML**: All necessary CSS, JavaScript, and the visualization logic are embedded directly into the `neural_tomography_scanner.html` file.

## Usage

1.  Download the `neural_tomography_scanner.html` file.
2.  Open it in a modern web browser (Chrome, Firefox, Safari, Edge).

### Controls

-   **3D View**:
    -   **Rotate**: Left-click and drag.
    -   **Zoom**: Scroll wheel.
    -   **Pan**: Middle-click and drag.
-   **Slice Views**:
    -   **Scroll Slices**: Use the sliders in the right panel or hover over a slice and use the scroll wheel.
-   **Windowing (Brightness/Contrast)**:
    -   **Adjust**: Right-click and drag anywhere on the page (left/right for width, up/down for center/level).
    -   **Presets**: Use the buttons in the top toolbar.

## Project Structure

```
/neural_tomography
├── neural_tomography_scanner.html  # The final, self-contained application
├── prepare_volume_data.py          # (Optional) Script to load real GPT-2 weights and generate data
├── embed_data.py                   # (Optional) Script to embed real data into the HTML
├── LICENSE                         # MIT License
└── README.md                       # This file
```

## Development

While the main HTML file is self-contained, the repository includes the Python scripts used to work with the real GPT-2 model data.

**Requirements**: `numpy`, `torch`, `transformers`, `scipy`

1.  **Generate Volume Data**: Run `python prepare_volume_data.py` to download the GPT-2 model and create a `volume_data.json` file.
2.  **Embed Data**: Run `python embed_data.py` to inject the data from `volume_data.json` into the HTML template, creating a final distributable file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
