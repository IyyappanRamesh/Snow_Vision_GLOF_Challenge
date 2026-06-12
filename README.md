# GLOFeagles '26: Hybrid CNN-Transformer Framework for Glacial Lake Detection

This repository contains Team SNOW_VISION's official submission for the GLOFeagles '26 Glacial Lake Detection Challenge as a part of NCVPRIPG 2206. The solution implements a data-efficient, semi-supervised TransUNetHybrid architecture designed to detect and segment highly volatile glacial lakes across complex high-altitude alpine terrains.

Repository Structure
Source Code/
├── 📄 model_architecture.py  # Network skeletal definition (ResNet-50 + Transformer Bridge)
├── 📄 utils.py               # Custom Hybrid Loss (BCE + Dice) and Dataset Loaders
├── 📄 train.py               # Active learning execution and model optimization loop
├── 📄 inference.py           # Evaluation pipeline for batch processing challenge datasets
├── 📄 requirements.py        # Complete environment library dependencies 
└── 📄 README.md              # Documentation and execution manual (This file)

The explanation video of the methodology and results are at https://youtu.be/Wo0h97mKXPo?si=Z8ZeFAE7_2ZQfHN6
