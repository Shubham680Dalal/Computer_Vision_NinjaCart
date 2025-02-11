# Computer_Vision_NinjaCart

# Problem Statement  
Ninjacart, India's largest fresh produce supply chain company, aims to develop an automated system for classifying vegetables in market images. The goal is to build a robust multiclass classifier capable of distinguishing between **onions, potatoes, tomatoes, and general market scenes (noise).** This will aid in streamlining inventory management and quality control using **computer vision techniques**.  

---

## Training Setup  
- Utilized **TPU with 8 replicas** for accelerated training.  
- Set **batch size = 128** for efficient parallel processing.  
- Handled **multiclass classification (4 classes):** `["tomato", "potato", "onion", "Indian market"]`.  

---

## Exploratory Data Analysis (EDA)  
- **Format Differences:** All **tomato images** were `.png`, while most other classes were in `.jpeg` or `.jpg`.  
- **Image Size Patterns:**  
  - **Tomato images** were typically `400-600×200-400` or `200-400×400-600`.  
  - **Onion, potato, and market images** were mostly `0-200×200-400`.  

---

## Data Preprocessing & Augmentation  
- Classes were **almost balanced**, but **underrepresented classes** were **augmented** to create a fully balanced dataset.  
- **Final training dataset:** **2,936 images** (`734 per class`), up from **2,508**.  

---

## Model Training  
Trained for **10 epochs** using the following models:  
✅ **AlexNet** *(trained from scratch)*  
✅ **VGG16** *(transfer learning)*  
✅ **ResNet** *(transfer learning)*  
✅ **MobileNet** *(transfer learning)*  

**Best Model Observed:**  
- **MobileNet**  
  - **Test Accuracy:** `89.46%`  
  - **Test Precision:** `90.49%`  
  - **Test Recall:** `89.46%`  

---

## Model Interpretation  
- **Used Grad-CAM and LIME** to highlight **regions influencing model predictions**.  
- **Identified** positive and negative regions that impacted classification decisions.  

---

## Key Observations & Takeaways  
1. **Tomato images had the highest accuracy** compared to other classes, indicating a **bias in generalization**.  
2. **Image format distribution is uneven:**  
   - **Tomato images:** Predominantly in `.png` format (high quality, lossless).  
   - **Other classes:** Mostly in `.jpeg/.jpg` (compressed, lossy).  
3. **Image size distribution is imbalanced:**  
   - **Tomato images were larger** (`400×600` and above), preserving more details.  
   - **Other classes had smaller images** (`0-200 × 200-400`), potentially **losing details during resizing**.  
4. **AlexNet (trained from scratch) performed well** because it **learned domain-specific features**.  
5. **Pre-trained models (VGG/ResNet) struggled**, likely due to **feature mismatches with ImageNet-trained filters**.  
6. **MobileNet outperformed VGG/ResNet** due to its **efficient feature extraction** and ability to generalize with **fewer parameters and a smaller dataset**.  

---

## Recommendations for Improvement  
✅ **Standardize Image Format:** Convert all images to `.png` before training to **eliminate compression-related biases**.  
✅ **Fine-Tune Transfer Learning Models:**  
   - Unfreeze more layers (**top 10-15**) instead of training only the Softmax layer.  
   - Use a **lower learning rate** for gradual adaptation.  
   - **Train for more epochs** to improve learning.  
✅ **Experiment with Data Augmentation:**  
   - **Upsample small images** to retain important features.  
   - **Use contrast enhancement** to compensate for image quality variations.  

---
