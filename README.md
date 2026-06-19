# GLOFeagles '26: Hybrid CNN-Transformer Framework for Glacial Lake Detection

This repository presents an end-to-end deep learning solution developed for the GLOFeagles 2026 challenge, targeting accurate automated segmentation of high-altitude glacial lakes to monitor GLOF flood risks.

To overcome severe land-to-water class imbalances and false positives caused by mountain terrain shadows, we implement a custom TransUNetHybrid architecture. The network features a robust ResNet-50 convolutional encoder to capture fine-grained spatial boundaries and sharp shorelines, a Vision Transformer (ViT) bridge utilizing multi-head self-attention to model landscape-wide global context, and a cascaded decoder with U-Net style skip-connections for precise localization recovery.

Trained via a curriculum-driven active learning loop using a unified Binary Cross-Entropy and Dice loss framework, the model achieves a Pixel Accuracy of 99.69%. To rigorously validate performance beyond pixel accuracy on highly imbalanced remote sensing data, the framework achieves a mean Intersection over Union (mIoU) of 77.20%, mean precision of 84.01%, mean recall of 84.08%, mean F1 score of 83.16%, and a Cohen's Kappa coefficient of 0.8302. These evaluation metrics confirm strong statistical agreement with ground-truth expert annotations, successfully differentiating turbid water bodies from complex alpine topography.

The explanation video of the methodology and results are at https://youtu.be/Wo0h97mKXPo?si=Z8ZeFAE7_2ZQfHN6
---

##  Repository Structure

```text
.
├── Evaluation on Validation Dataset/
│   ├── Validation_Predictions/
│   ├── Snow_Vision_Validation_Report.pdf
│   └── TEST_DATASET_INFERENCE_CODE.ipynb
│
├── GLOFeagles_2026_segmented_masks/
│
├── Source Code/
│   ├── model_architecture.py
│   ├── utils.py
│   ├── train.py
│   ├── inference.py
│   └── requirements.py
│
├── README.md
├── Snow_Vision_Technical_&_Evalaution_Report.pdf
├── Trained_Iterative_Unsupervised_Learning_Model_code.ipynb
└── best_finetuned_model.pt


