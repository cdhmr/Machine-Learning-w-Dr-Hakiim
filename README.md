# Machine-Learning-w-Dr-Hakiim
ML Quiz : Quality-Aware Convolutional Neural Network for Glaucoma Detection

# Quality-Aware CNN for Glaucoma Detection on HYGD

This repository contains the code and technical report for a deep learning pipeline designed to detect Glaucomatous Optic Neuropathy (GON) using the Hillel Yaffe Glaucoma Dataset (HYGD). 

The project evaluates a standard CNN baseline against a Quality-Aware CNN that utilizes a custom `QualityWeightedBCELoss` function. It also implements an automated OpenCV-based optic disc localization and cropping pipeline to mitigate shortcut learning.

## Repository Structure

* `glaucoma_detection.ipynb`: The main Jupyter Notebook containing all steps from data preprocessing, patient-level splitting, model architectures, training loops, threshold tuning, to Grad-CAM evaluation.
* `MAA62052_Group 3_CNN_HYGD.pdf`: The final technical report detailing the methodology, results, and clinical considerations.
* `README.md`: This documentation file.

## Requirements

The code is written in Python. Ensure you have the following libraries installed before running the notebook:

* torch
* torchvision
* opencv-python (cv2)
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* Pillow
* tqdm

You can install the dependencies using pip:
`pip install torch torchvision opencv-python pandas numpy scikit-learn matplotlib seaborn Pillow tqdm`

## Dataset Preparation

This project uses the Hillel Yaffe Glaucoma Dataset (HYGD). Due to medical data licensing, the dataset is not included in this repository.

1. Download the dataset (v1.0.0) from PhysioNet: [HYGD on PhysioNet](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.0.0/)
2. Extract the downloaded files.
3. Ensure the `Labels.csv` file and the folder containing the raw images are placed in the appropriate directory as specified in the first few cells of `glaucoma_detection.ipynb` (update the `IMAGE_DIR` and `CSV_PATH` variables in the notebook if necessary).

## Reproduction Instructions

To reproduce the results, models, and evaluation metrics presented in the report:

1. Clone this repository to your local machine.
2. Ensure the HYGD dataset is downloaded and paths are correctly set in the notebook.
3. Open `glaucoma_detection.ipynb` using Jupyter Notebook, JupyterLab, or Google Colab.
4. Run the notebook sequentially from the first cell to the last (`Restart & Run All`).

The notebook will automatically:
* Perform the 70/15/15 patient-level data split.
* Train the Baseline CNN.
* Train the Quality-Aware CNN.
* Execute the offline preprocessing script to crop the optic discs and save them to a `Cropped_Images/` directory.
* Train the Cropped-Image CNN.
* Train the transfer-learning ResNet-18 model.
* Generate all metric tables and Grad-CAM visualizations.

## Key Results

* **Baseline CNN:** High mathematical performance (AUC 0.977) but highly susceptible to shortcut learning based on image background artifacts.
* **Quality-Aware CNN:** Custom loss weighting improved sensitivity (0.975) while acknowledging image unreliability.
* **Cropped-Image CNN:** Optic-disc isolation stabilized performance across varying image qualities and removed background bias, demonstrating more clinically valid feature extraction.
