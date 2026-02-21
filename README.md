# Swin2SR: Super-Resolution Guided Surveillance Enhancement

A proactive deep learning system designed to transform passive surveillance into an intelligent security mechanism. This module utilizes **SwinV2 Transformers** to enhance low-quality surveillance patches (ROI) for robust soft-biometric extraction.

##  Key Features
- **Swin-Transformer Architecture:** Uses Shifted-Window Self-Attention for superior texture reconstruction in blurry CCTV frames.
- **Compressed-SR Task:** Specifically optimized to remove digital artifacts and JPEG noise common in security footage.
- **Biometric Recovery:** Enhances pixel-density of shoes, attire, and accessories to improve detection accuracy by subsequent AI models (YOLOv8).
- **Patch-Based ROI Logic:** Selective enhancement of person-of-interest crops to ensure real-time performance.

##  Tech Stack
- **Framework:** PyTorch
- **Models:** Swin2SR (SwinV2 Transformer)
- **Libraries:** OpenCV, TIMM, NumPy, Matplotlib

##  Project Structure
- `models/`: Contains the Swin2SR Transformer architecture.
- `run_sr.py`: Main execution script for processing low-res inputs.
- `data/`: Sample input crops and enhanced outputs.
- `weights/`: Pre-trained model weights (to be downloaded separately).

##  License
This project is licensed under the MIT License.

Swin2SR_Project/
├── dataset/             # (Local Only) Contains DIV2K/Surveillance training data
├── inputs/              # Low-resolution source images (e.g., image1.jpg)
├── models/              # Core architecture logic
│   ├── __init__.py      # Package initialization
│   └── network_swins2sr.py # SwinV2 Transformer implementation
├── results/             # SR outputs (e.g., enhanced_image1.png)
├── weights/             # (Local Only) Pre-trained model weights (.pth)
├── .gitignore           # Instructions to exclude heavy files from Git
├── LICENSE              # MIT License
├── README.md            # Project documentation
└── run_sr.py            # Main execution script for inference
