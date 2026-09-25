#  Solar Panel Defect Classification

## 📋 Project Summary & Objective

Solar panels lose efficiency when their surfaces are obstructed or damaged — by dust, bird droppings, snow, or physical/electrical faults. Manual inspection of large solar farms is slow, expensive, and error-prone.

This project builds an **automated image classification pipeline** that detects and categorizes the condition of solar panels from photographs, enabling faster, data-driven maintenance decisions. It walks through the full model-development lifecycle:

1. A **baseline CNN** trained from scratch to establish a performance floor.
2. **Transfer learning** using pretrained ImageNet backbones (MobileNetV2, EfficientNetB0) to boost accuracy and reduce training time.
3. **Hyperparameter Optimization (HPO)** with Keras Tuner to systematically search for the best-performing configuration.
4. A **comparative evaluation** of all trained architectures to identify the most production-ready model.

The end goal is a lightweight, high-accuracy classifier suitable for deployment in solar-farm monitoring pipelines (e.g., drone or fixed-camera inspection systems).


---
🌐 **Live Web Application:** [Click here to view the Deployed App](https://your-deployment-link.com)
---


## Dataset Description

| Attribute | Details |
| :--- | :--- |
| **Source** | `https://www.kaggle.com/datasets/salonipandagale/solar-panel-defect-classification-dl-project` |
| **Total Images** | 885 |
| **Classes (6)** | `Bird-drop`, `Clean`, `Dusty`, `Electrical-damage`, `Physical-Damage`, `Snow-Covered` |
| **Train / Validation Split** | 80% / 20% (708 training images / 177 validation images) |
| **Image Size** | 224 × 224 px (RGB) |
| **Batch Size** | 32 |
| **Random Seed** | 42 |
| **Class Balance** | Imbalanced — handled via computed **class weights** during training (e.g., `Physical-Damage` and `Electrical-damage` are minority classes) |


## Model Architectures & Methodology

### 1. Base Model (CNN from Scratch)
A simple custom Convolutional Neural Network built with stacked `Conv2D` + `MaxPooling2D` blocks, `Rescaling` normalization, and a dense classification head. Used as a performance baseline. An improved variant adds `BatchNormalization` and `Dropout` layers to combat overfitting observed in early training runs.

### 2. Transfer Learning
Pretrained **ImageNet** backbones are used as frozen feature extractors, with a custom classification head (`GlobalAveragePooling2D` → `Dense(ReLU)` → `Dense(Softmax)`) trained on top:
- **MobileNetV2** — a lightweight, mobile-friendly architecture using inverted residual blocks and depthwise separable convolutions.
- **EfficientNetB0** — a compound-scaled architecture balancing network depth, width, and resolution for strong accuracy-to-compute efficiency.

Data augmentation (`RandomFlip`, `RandomRotation`, `RandomZoom`) is applied on the fly to improve generalization, and **class weighting** is used to counteract dataset imbalance.

### 3. Hyperparameter Optimization (HPO)
`keras-tuner`'s **RandomSearch** is used to tune the EfficientNetB0-based pipeline over:
- Data augmentation strength (`rotation_factor`, `zoom_factor`)
- `dropout_rate`
- Dense layer width (`dense_units`)
- Learning rate (log-scaled search)

`EarlyStopping` (monitoring `val_accuracy`, patience = 5) is used during the search to avoid wasted compute on unpromising trials, and the best model is evaluated on the held-out validation set.

---

## Model Comparison & Performance Evaluation

| Model Architecture | Epochs / Setup | Training Accuracy (%) | Validation Accuracy (%) |
| :--- | :---: | :---: | :---: |
| **Base CNN** | 10 | 97 | 63 |
| **Base CNN** | 20 | 99 | 65 |
| **Base CNN + BatchNorm/Dropout** | 10 | 86 | 14 |
| **MobileNetV2** | 10 | 95 | 76 |
| **MobileNetV2** | 20 | 92 | 68 |
| **EfficientNetB0** | 15 | 94 | 80 |
| **EfficientNetB0** | HPO (Keras Tuner, 20 trials) | 91 | 82 |



---

## Installation & Setup

### Prerequisites
- Python 3.9+
- A GPU-enabled environment is **strongly recommended** (the notebook was developed on Google Colab with a T4 GPU)
- Google Drive access (if running on Colab with the dataset stored in Drive) *or* a local copy of the dataset

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/solar-panel-defect-classification.git
cd solar-panel-defect-classification
```

### 2. Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install tensorflow matplotlib keras-tuner jupyter
```

Or, using a `requirements.txt`:
```
tensorflow>=2.15
matplotlib
keras-tuner
jupyter
```
```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset
Organize your dataset in the following structure (one subfolder per class), then update the `DATASET_DIR` path in the notebook:
```
Solar_Panel_Dataset/
├── Bird-drop/
├── Clean/
├── Dusty/
├── Electrical-damage/
├── Physical-Damage/
└── Snow-Covered/
```

### 5. Run the notebook
```bash
jupyter notebook Solar_Panel_Defect_Classification.ipynb
```
If running on **Google Colab**, mount your Google Drive when prompted and ensure `DATASET_DIR` points to the correct path (e.g., `/content/drive/MyDrive/Solar_Panel_Dataset`).

---

## Key Findings & Conclusion

- The **baseline CNN** overfit quickly — training accuracy climbed to ~97–100%, but validation accuracy plateaued around 63–66%, and the BatchNorm/Dropout variant intended to curb overfitting instead collapsed to 14% validation accuracy, indicating that regularization choice/configuration needs further tuning for this small (885-image) dataset.
- **Transfer learning** closed much of that generalization gap: MobileNetV2 reached 76% validation accuracy in its first 10 epochs, and EfficientNetB0 reached 80% validation accuracy in 15 epochs — both clear improvements over the from-scratch baseline.
- Continuing training beyond the first run (MobileNetV2 and the base CNN both trained for additional epochs on top of their prior weights) **did not** improve validation accuracy further and in MobileNetV2's case reduced it (76% → 68%), suggesting the frozen-backbone models had begun overfitting the small training set.
- **Hyperparameter Optimization** (Keras Tuner `RandomSearch` over augmentation strength, dropout, dense units, and learning rate on EfficientNetB0) produced the **best validation accuracy in the notebook: ~82%**, ahead of any single non-tuned run.
- Class imbalance (particularly for `Electrical-damage` and `Physical-Damage`) was addressed via computed class weights rather than oversampling, helping the model avoid bias toward majority classes like `Clean` and `Dusty`.
- **Next steps:** apply early stopping / more conservative epoch counts to the non-HPO transfer-learning runs to avoid the overfitting seen on continued training; benchmark additional architectures (e.g., Xception, InceptionV3, ResNet50, VGG16); expand the dataset for more robust minority-class performance; and evaluate inference latency/model size trade-offs for edge or drone deployment.

---

### 6. References & Citations

1. **MobileNetV2:**
   > Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 4510–4520).

2. **EfficientNet-B0 (Base Architecture):**
   > Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. In Proceedings of the 36th International Conference on Machine Learning (ICML) (pp. 6105–6114).
---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

