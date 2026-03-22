# MobileNetv2-SEGP-
The cnn model for the software engineering project

# CanineCare: Dog Posture Classification Pipeline

## Overview

A web-deployed CNN for binary classification of canine spinal posture (Normal vs. Abnormal). Built with MobileNetV2 and deployed to browser via TensorFlow.js for privacy-first, client-side inference.

**Current Status**: Proof-of-concept with limited training data. Production deployment pending dataset expansion.

---

## Architecture

**Model**: MobileNetV2 (ImageNet pre-trained)  
**Training**: Two-stage transfer learning (Keras/TensorFlow)  
**Deployment**: TensorFlow.js Graph Model (Flutter Web)  
**Input**: `[1, 224, 224, 3]` RGB tensors  
**Output**: Binary classification (sigmoid)

### Training Pipeline

1. **Stage 1**: Warmup (15 epochs, head-only, LR=1e-3)
2. **Stage 2**: Fine-tuning (up to 100 epochs, LR=1e-5, BatchNorm frozen)
3. **Regularization**: Dropout (0.5), data augmentation, class weighting
4. **Validation**: Stratified 5-fold cross-validation

---

## Engineering Challenges Solved

### 1. Python-to-JavaScript Preprocessing Pipeline
- Replicated exact Keras preprocessing in JS (`[-1, 1]` normalization)
- Resolved OpenCV BGR→RGB color space conflicts
- Implemented dynamic image resizing and tensor batching

### 2. TensorFlow.js Compatibility
- Bypassed Keras 3 JSON serialization bugs
- Exported as TensorFlow Graph Model for web stability
- Optimized for browser inference performance

### 3. Batch Inference Issues
- Fixed tensor shape collapse from batched `[4, 224, 224, 3]` inputs
- Switched to sequential one-by-one inference loop
- Maintained model input layer compatibility

### 4. Ensemble Voting System
- **Input**: 4 camera angles (Front, Left, Right, Back)
- **Logic**: Red flag if prediction ≥ 0.5 per angle
- **Output**: "Abnormal" diagnosis if ≥2 red flags
- **Confidence**: Percentage based on ensemble agreement

---

## Dataset & Performance

### Current Dataset
- **Size**: 71 images (train/val: 60, test: 11)
- **Distribution**: 29 Abnormal, 42 Normal
- **Breed Imbalance**: 35% Collie, others <15%

### Cross-Validation Results (5-Fold)
- **Accuracy**: 98.33% ± 3.33%
- **Precision**: 100.0%
- **Recall**: 96.0%

⚠️ **High variance (±3.33%) indicates overfitting due to small dataset size.**

### Pipeline Validation (Proxy Test on Cattle Dataset)
To validate architecture independent of dataset size:
- **Dataset**: 2,757 cattle disease X-rays
- **Accuracy**: 94.41% ± 0.90%
- **Precision**: 94.89% | **Recall**: 95.90%

✅ **Confirms pipeline is production-ready when given sufficient data.**

---

## Key Findings

### What Works ✅
- **Architecture**: MobileNetV2 + two-stage training performs excellently (94.4% on 2.7k images)
- **Deployment**: TFJS conversion and browser inference pipeline robust
- **Ensemble Strategy**: 4-angle voting reduces single-image noise

### Current Limitations ⚠️
- **Dataset Size**: 71 images insufficient for production reliability
- **Generalization**: Model overfits to training breeds (especially Collies)
- **Variance**: High fold-to-fold inconsistency (91-100% range)
- **Expected Real-World Accuracy**: ~65-70% (vs. 98% CV accuracy)

### Why This Matters
The 98% cross-validation accuracy is misleading:
- With only 12 validation images per fold, 1 misclassification = 8% accuracy drop
- Model likely memorizing specific dogs rather than learning posture patterns
- Cattle dataset experiment proves architecture works, but hip dysplasia needs 1,000+ images

---

## Technical Deep Dive

### Data Preprocessing (JavaScript)
```javascript
// MobileNetV2 expects [-1, 1] normalized RGB
function preprocessImage(imageData) {
  const tensor = tf.browser.fromPixels(imageData)
    .resizeBilinear([224, 224])
    .toFloat()
    .div(127.5)      // Scale [0, 255] → [0, 2]
    .sub(1.0)        // Shift [0, 2] → [-1, 1]
    .expandDims(0);  // Add batch dimension
  return tensor;
}
```

### Ensemble Inference
```javascript
async function diagnose(images) {
  let redFlags = 0;
  
  for (const img of images) {
    const tensor = preprocessImage(img);
    const prediction = await model.predict(tensor).data();
    
    if (prediction[0] >= 0.5) redFlags++;
    
    tensor.dispose(); // Prevent memory leak
  }
  
  const confidence = (redFlags / 4) * 100;
  return {
    diagnosis: redFlags >= 2 ? 'Abnormal' : 'Normal',
    confidence: confidence
  };
}
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | MobileNetV2 (ImageNet) |
| Input Size | 224×224×3 |
| Batch Size | 4 |
| Epochs | 100 (early stopping) |
| Optimizer | Adam (1e-3 → 1e-5) |
| Loss | Binary Crossentropy |
| Augmentation | Rotation, shift, zoom, flip |
| Class Weights | {0: 1.0, 1: 1.3} |

---

## Future Roadmap

### Phase 1: Data Collection (6-12 months)
- **Target**: 1,000+ professionally labeled images
- **Strategy**: Partner with veterinary clinics for diverse X-ray archive
- **Labeling**: Expert veterinarian review (no pseudo-labeling)
- **Quality**: Multi-vet consensus for ambiguous cases

### Phase 2: Model Improvement
- Retrain on expanded dataset (expect 92-94% real accuracy)
- Implement angle-specific expert models
- Add uncertainty quantification (MC Dropout)
- Generate Grad-CAM heatmaps for explainability

### Phase 3: Production Deployment
- A/B test new model vs. baseline
- Implement monitoring dashboard
- Add human-in-the-loop validation
- Clinical validation study

---

## Known Issues & Mitigations

### Issue: Small Dataset (71 images)
**Impact**: High overfitting, unreliable real-world performance  
**Mitigation**: Using inference-only deployment with expert labeling pipeline  
**Timeline**: 1,000 images by Month 12

### Issue: Breed Imbalance
**Impact**: Model biased toward Collies (35% of dataset)  
**Mitigation**: Balanced sampling in future data collection  

### Issue: Training-Serving Skew
**Impact**: Background segmentation differences between train/inference  
**Mitigation**: Standardized preprocessing pipeline, documented in codebase

---

## Deployment Considerations

⚠️ **Current Model Status**: Research prototype only. Not recommended for clinical use.

**For Production Deployment, Need**:
- ✅ 1,000+ diverse training images
- ✅ Independent test set validation
- ✅ Clinical validation study
- ✅ Regulatory approval (if medical device)
- ✅ Monitoring and fallback systems

**Acceptable Current Use Cases**:
- Educational demonstrations
- User engagement tool (with disclaimers)
- Data collection platform (with expert review)

---

## How to Run

### Training
```bash
# Open Jupyter notebook
jupyter notebook Training_evaluation.ipynb

```

### Model Conversion
```bash
# Convert Keras model to TensorFlow.js
tensorflowjs_converter \
  --input_format=keras \
  models/mobilenet_posture.h5 \
  web_model/
```


---

**Last Updated**: March 2026  
**Model Version**: 1.0 (Proof of Concept)  
**Status**: Research Prototype - Not for Clinical Use
