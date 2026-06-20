# Machine-Learning-w-Dr-Hakiim
ML Quiz : Quality-Aware Convolutional Neural Network for Glaucoma Detection

```markdown
# Quality-Aware Convolutional Neural Network for Glaucoma Detection

This repository contains the code, models, and technical report for the detection of Glaucomatous Optic Neuropathy (GON) using the Hillel Yaffe Glaucoma Dataset (HYGD). The project explores the impact of image quality on classification performance through a custom Quality-Aware CNN and an automated Optic Disc localization pipeline.

## Authors
* Achmad Euro Winrasaputra (235091001111010)
* Kharisma Jaya Abdallah (235091000111008)
* Alfian Rafid Rabbani (235091001111011)
* Marwan Hadid Ramdhani (235091007111003)

**Course:** MAA62052  
**Academic Session:** 2025/2026-2  

## Dataset
The project utilizes the **Hillel Yaffe Glaucoma Dataset (HYGD)**. 
* **Source:** [PhysioNet HYGD v1.0.0](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.0.0/)
* **Structure:** 747 fundus images from 288 patients.
* **Citation:** Abramovich, O., Pizem, H., Fhima, J., Berkowitz, E., Gofrit, B., Van Eijgen, J., Blumenthal, E., & Behar, J. (2025). *Hillel Yaffe Glaucoma Dataset (HYGD): A Gold-Standard Annotated Fundus Dataset for Glaucoma Detection* (v1.0.0). PhysioNet. https://doi.org/10.13026/z0akkm33

## Features & Methodology
1. **Patient-Level Splitting:** GroupShuffleSplit ensures zero data leakage between training, validation, and test sets.
2. **Optic Disc Localization:** A custom OpenCV pipeline using green-channel extraction, morphological black-hat filtering, and heatmap multiplication to isolate the optic disc.
3. **Quality-Aware Training:** Implementation of a custom `QualityWeightedBCELoss` that scales the loss gradient based on the dataset's provided quality scores.
4. **Transfer Learning Baseline:** ResNet-18 implemented as a comparative baseline.
5. **Interpretability:** Grad-CAM integration for visualizing model focal points (Optic Cup/Disc vs. artifacts).

## Prerequisites
Ensure you have Python 3.8+ installed. The required libraries are:

```bash
pip install torch torchvision
pip install opencv-python pillow
pip install pandas numpy scikit-learn
pip install matplotlib seaborn

```

## Repository Structure

```text
.
├── data/
│   ├── images/               # Raw HYGD images (Download from PhysioNet)
│   ├── Cropped_Images/       # Auto-generated cropped images
│   └── Labels.csv            # Ground truth and quality scores
├── figs/                     # Output figures (Grad-CAM, ROC, Confusion Matrices)
├── models/                   # Saved model weights (.pth)
├── src/
│   ├── dataset.py            # Custom PyTorch Dataset classes and transforms
│   ├── preprocessing.py      # Optic disc cropping and heatmap generation
│   ├── architecture.py       # Custom CNN and Quality-Aware CNN definitions
│   ├── training.py           # Training loops and QualityWeightedBCELoss
│   └── evaluation.py         # Metrics calculation and Grad-CAM visualization
├── main.ipynb                # Main execution notebook
├── MAA62052_Group_3_CNN_HYGD.pdf # Final Technical Report
└── README.md

```

## Reproduction Instructions

### 1. Data Preparation

1. Download the dataset from PhysioNet and extract the images into `data/images/`.
2. Place `Labels.csv` in the `data/` directory.

### 2. Execution

The entire pipeline can be executed sequentially via `main.ipynb`. Alternatively, if running scripts:

1. **Preprocess Images (Optional but recommended):**
Run the cropping pipeline to generate optic-disc centered images.
```python
python src/preprocessing.py --source data/images --target data/Cropped_Images

```


2. **Train Models:**
Train the baseline, quality-aware, and ResNet-18 models. Weights will be saved in `models/`.
```python
python src/training.py

```


3. **Evaluate and Visualize:**
Generate the metrics table (Accuracy, Precision, Recall, Specificity, F1, AUC) and Grad-CAM outputs.
```python
python src/evaluation.py

```



## Results Summary

* **Baseline CNN:** F1: 0.962 | AUC: 0.977
* **Quality-Aware CNN:** F1: 0.963 | AUC: 0.957
* **Cropped-Image CNN:** F1: 0.927 | AUC: 0.928
* **ResNet-18:** F1: 0.947 | AUC: 0.977

Detailed quality-stratified metrics and error analysis can be found in the technical report PDF included in this repository.

```

```
