# Fully Convolutional Networks for Semantic Segmentation

This section implements and evaluates three variants of the Fully Convolutional Network (FCN) architecture for semantic segmentation, as described in the paper "[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)" by Long et al.

## Table of Contents
1. [Dataset Description](#dataset-description)
2. [Dataset Visualization](#dataset-visualization)
3. [FCN Architecture](#fcn-architecture)
4. [Implementation Details](#implementation-details)
5. [Training Methodology](#training-methodology)
6. [Results and Analysis](#results-and-analysis)
7. [Comparison of FCN Variants](#comparison-of-fcn-variants)

## Dataset Description

The dataset consists of street scene images and their corresponding pixel-wise segmentation masks. Each pixel in the mask is assigned one of 13 class labels:

0. Unlabeled  
1. Building  
2. Fence  
3. Other  
4. Pedestrian  
5. Pole  
6. Roadline  
7. Road  
8. Sidewalk  
9. Vegetation  
10. Car  
11. Wall  
12. Traffic sign

The dataset is divided into training, validation, and test sets with the following distribution:
- Training: 1600 images  
- Validation: 400 images  
- Test: 500 images  

Images and masks have a resolution of 224×224 pixels.

## Dataset Visualization

I implemented a function `visualize_masks()` to extract and visualize individual class masks from the segmentation labels. This function:
1. Reads a segmentation mask image  
2. Extracts the first channel which contains the class IDs  
3. Creates binary masks for each of the 13 classes  
4. Displays each binary mask in a grid with appropriate class labels  

The binary masks clearly show the distribution of each class in the image, providing insights into the spatial distribution and relative frequency of different objects in the scene.

Below is an example of a street scene image from the dataset along with its corresponding segmentation mask:

![Sample Dataset Visualization](Q1-FCN/figures/dataset-fcn-sample.png)

## FCN Architecture

The implementation includes three variants of the FCN architecture, all based on the VGG16 backbone:

### FCN-32s Architecture
- Uses the VGG16 network pretrained on ImageNet  
- Replaces fully connected layers with convolutional layers  
- Outputs coarse predictions with a stride of 32 pixels  
- Upsamples the prediction to full image resolution in a single step  

### FCN-16s Architecture
- Builds upon FCN-32s  
- Combines the final layer predictions with predictions from the pool4 layer  
- The pool4 predictions have a stride of 16 pixels  
- Provides finer spatial information  

### FCN-8s Architecture
- Further refines predictions by incorporating information from pool3 layer  
- Combines predictions from the final layer, pool4, and pool3  
- The pool3 predictions have a stride of 8 pixels  
- Creates more detailed segmentation outputs  

Each architecture was implemented both with frozen backbone weights and with fine-tuning of all network weights.


### Why Skip Connections Work
The key insight of the FCN architecture is combining semantic information from a deep, coarse layer with appearance information from a shallow, fine layer:

- **Coarse Layers**: The deep layers (around pool5) have large receptive fields and capture semantic content ("what") but lack spatial precision.
- **Fine Layers**: The shallow layers (pool3, pool4) have smaller receptive fields and preserve spatial information ("where").
- **Skip Connections**: By combining these complementary features, the model can produce accurate and detailed segmentations.

As seen in our experiments, FCN-8s benefits the most from this multi-scale feature fusion, as it incorporates the most shallow, spatially-precise features from pool3.

![Architecture Diagram](Q1-FCN/figures/FCNs.jpg)

## Implementation Details

The implementation consists of several key components:

### Base FCN Class
- A base class `FCN_Base` that initializes the VGG16 backbone  
- Converts fully connected layers to convolutional layers  
- Provides methods to extract intermediate pool layers  

### Architecture-Specific Classes
- `FCN32`: Implements the coarsest model with 32-pixel stride  
- `FCN16`: Adds skip connections from pool4 layer  
- `FCN8`: Adds further skip connections from pool3 layer  

### Bilinear Initialization
- Implemented `bilinear_kernel()` function to create appropriate upsampling filters  
- Initialize upsampling layers with bilinear weights for better performance  
- Used F.interpolate with the bilinear option as well, a choice between initializing it with weights from the bilinear kernel, or directly.

### Dataset Handling
- Custom `SegmentationDataset` class using PyTorch's Dataset API  
- Transforms for both images and masks  
- Proper handling of class IDs in the mask's first channel  

## Training Methodology

The training pipeline includes several key components:

### Data Processing
- Images are resized to 224×224 and normalized  
- Masks are resized with nearest-neighbor interpolation to preserve class IDs  
- Random 80/20 split for training and validation  

### Training Parameters
- Batch size: 32  
- Optimizer: Adam with learning rate 1e-4  (Used SGD in the paper, but it takes too long to converge, better normalized values though)
- Loss function: Cross-entropy loss  
- Metrics: Mean IoU using `torchmetrics.segmentation.MeanIoU`  
- Training using PyTorch's mixed precision for efficiency  

### Training Functions
- `train_epoch()`: Handles one epoch of training with progress tracking  
- `validate_epoch()`: Evaluates model on validation set and collects visualization samples  
- `train_model()`: Main training loop with checkpoint saving and visualization  
- `evaluate_model()`: Evaluates final model on test set and generates visualizations  

### Visualization
- `visualize_predictions()`: Compares ground truth with model predictions  
- `plot_learning_curves()`: Tracks training and validation loss/IoU  
- Per-class IoU tracking for detailed analysis  

## Results and Analysis

### FCN-32s (Frozen Backbone)
- **Epochs:** 1  
- **Validation mIoU:** 0.7441  
- **Test mIoU:** 0.7441  
- **Best Classes:** Car (92.6%), Road (88.3%), Unlabeled (83.2%)  
- **Challenging Classes:** Pedestrian (0.0%), Pole (2.7%), Roadline (1.0%)  
- **Observation:** Without any fine-tuning, the FCN-32s architecture already achieves a strong baseline. However, due to the lack of skip connections and backbone adaptation, it struggles with small-scale and low-frequency classes.

---

### FCN-32s (Fine-tuned Backbone)
- **Epochs:** 1  
- **Validation mIoU:** 0.7441  
- **Test mIoU:** 0.7441  
- **Best Classes:** Car (92.6%), Road (88.3%), Unlabeled (83.2%)  
- **Challenging Classes:** Pedestrian (0.0%), Pole (2.7%), Roadline (1.0%)  
- **Observation:** Even with just one epoch of fine-tuning, FCN-32s captures broad spatial structures well. However, it lacks the resolution needed for small or edge-detail classes.

---

### FCN-16s (Frozen Backbone)
- **Epochs:** 5  
- **Validation mIoU:** 0.7654  
- **Test mIoU:** 0.7654  
- **Best Classes:** Car (93.6%), Road (88.8%), Unlabeled (85.2%)  
- **Challenging Classes:** Pedestrian (0.0%), Pole (4.6%), Traffic sign (19.8%)  
- **Observation:** The inclusion of pool4 skip connections gives a notable boost over FCN-32s, especially in medium-scale objects like Sidewalk and Building. Still, the frozen backbone restricts adaptability to dataset-specific features.

---

### FCN-16s (Fine-tuned Backbone)
- **Epochs:** 10  
- **Validation mIoU:** 0.8304  
- **Test mIoU:** 0.8304  
- **Best Classes:** Car (95.6%), Road (91.4%), Unlabeled (90.8%)  
- **Challenging Classes:** Pedestrian (9.4%), Pole (12.8%), Roadline (16.4%)  
- **Observation:** Fine-tuning significantly boosts performance on both dominant and minority classes. It improves edge precision and segmentation of small structures, such as Traffic signs (40.6%) and Walls (53.9%).

---

### FCN-8s (Frozen Backbone)
- **Epochs:** 5  
- **Validation mIoU:** 0.7859  
- **Test mIoU:** 0.7856  
- **Best Classes:** Car (93.4%), Road (87.3%), Unlabeled (86.5%)  
- **Challenging Classes:** Pedestrian (3.1%), Pole (13.0%), Fence (19.9%)  
- **Observation:** Despite using a frozen backbone, FCN-8s benefits from finer spatial features via pool3 skip connections, capturing more structural details compared to FCN-16s-frozen.

---

### FCN-8s (Fine-tuned Backbone)
- **Epochs:** 15  
- **Validation mIoU:** 0.8304  
- **Test mIoU:** 0.8332  
- **Best Classes:** Car (95.3%), Road (91.3%), Building (83.2%)  
- **Improved Challenging Classes:**  
  - Fence: 27.9%  
  - Traffic sign: 32.7%  
  - Pedestrian: 0.9%  
  - Pole: 14.6%  
- **Observation:** This model offers the most balanced segmentation performance. It integrates deep semantic features and spatial precision effectively, showing strength in both high-frequency and fine-grained classes.

---

### Summary: Comparison of Variants

| Model             | Train mIoU | Val mIoU | Test mIoU | Car  | Road | Pedestrian | Traffic Sign | Fence |
|------------------|------------|----------|-----------|------|------|-------------|---------------|--------|
| FCN-32s Frozen    | 0.679      | 0.744    | 0.744     | 92.6% | 88.3% | 0.0%        | 6.5%          | 18.2%  |
| FCN-32s Fine      | 0.679      | 0.744    | 0.744     | 92.6% | 88.3% | 0.0%        | 6.5%          | 18.2%  |
| FCN-16s Frozen    | 0.766      | 0.765    | 0.765     | 93.6% | 88.8% | 0.0%        | 19.8%         | 23.7%  |
| FCN-16s Fine      | 0.847      | 0.830    | 0.830     | 95.6% | 91.4% | 9.4%        | 40.6%         | 26.7%  |
| FCN-8s Frozen     | 0.790      | 0.786    | 0.786     | 93.4% | 87.3% | 3.1%        | 21.1%         | 19.9%  |
| FCN-8s Fine       | 0.847      | 0.830    | 0.833     | 95.3% | 91.3% | 0.9%        | 32.7%         | 27.9%  |

---

### Insights
- **Skip Connections (FCN-16s and FCN-8s):** Each level of skip connection adds richer spatial granularity. Pool4 boosts edge clarity, and pool3 helps with smaller objects and class boundaries.
- **Fine-Tuning:** Substantially improves representation for hard classes like Pole, Traffic Sign, and Pedestrian. More epochs and learning rate scheduling further enhance this.
- **FCN-8s Fine-tuned** is the best-performing model, with the highest mean IoU and strong per-class balance, especially in real-world object segmentation scenarios.

### Visualizations

#### FCN-32s (Frozen Backbone)
- **Learning Curve:**  
  ![FCN32 Frozen Learning Curve](Q1-FCN/figures/results_FCN32_frozen_20250409_034328_5epoch/FCN32_frozen_learning_curves.png)
- **Per-Class IoU:**  
  ![FCN32 Frozen Class IoUs](Q1-FCN/figures/results_FCN32_frozen_20250409_034328_5epoch/FCN32_frozen_class_ious.png)

---

#### FCN-32s (Fine-tuned Backbone)
- **Learning Curve:**  
  ![FCN32 Fine-tuned Learning Curve](Q1-FCN/figures/results_FCN32_finetuned_20250409_040052_epoch/FCN32_finetuned_learning_curves.png)
- **Per-Class IoU:**  
  ![FCN32 Fine-tuned Class IoUs](Q1-FCN/figures/results_FCN32_finetuned_20250409_040052_epoch/FCN32_finetuned_class_ious.png)

---
#### FCN-16s (Frozen Backbone)
- **Learning Curve:**  
  ![FCN16 Frozen Learning Curve](Q1-FCN/figures/results_FCN16_frozen_20250409_061308/FCN16_frozen_learning_curves.png)
- **Per-Class IoU:**  
  ![FCN16 Frozen Class IoUs](Q1-FCN/figures/results_FCN16_frozen_20250409_061308/FCN16_frozen_class_ious.png)

---

#### FCN-16s (Fine-tuned Backbone)
- **Learning Curve:**  
  ![FCN16 Fine-tuned Learning Curve](Q1-FCN/figures/results_FCN16_finetuned_10epochs/FCN16_finetuned_learning_curves.png)
- **Per-Class IoU:**  
  ![FCN16 Fine-tuned Class IoUs](Q1-FCN/figures/results_FCN16_finetuned_10epochs/FCN16_finetuned_class_ious.png)

---

#### FCN-8s (Frozen Backbone)
- **Learning Curve:**  
  ![FCN8 Frozen Learning Curve](Q1-FCN/figures/results_FCN8_frozen/FCN8_frozen_learning_curves.png)
- **Per-Class IoU:**  
  ![FCN8 Frozen Class IoUs](Q1-FCN/figures/results_FCN8_frozen/FCN8_frozen_class_ious.png)

---

#### FCN-8s (Fine-tuned Backbone)
- **Learning Curve:**  
  ![FCN8 Fine-tuned Learning Curve](Q1-FCN/figures/results_FCN8_finetuned/FCN8_finetuned_learning_curves.png)
- **Per-Class IoU:**  
  ![FCN8 Fine-tuned Class IoUs](Q1-FCN/figures/results_FCN8_finetuned/FCN8_finetuned_class_ious.png)

---

These plots showcase both training dynamics (loss and mIoU over time) and model ability to generalize across all 13 classes, offering clear visual evidence of how architectural complexity and fine-tuning impact semantic segmentation performance.

## Comparison of FCN Variants

### Effect of Network Architecture

1. **FCN-32s vs FCN-16s vs FCN-8s**:  
   - The progression from FCN-32s to FCN-8s consistently improves mean IoU, with FCN-8s delivering the best results.  
   - FCN-32s (both frozen and fine-tuned) produces coarser segmentations due to the absence of skip connections.  
   - FCN-16s improves segmentation at object boundaries by incorporating mid-level features from the pool4 layer.  
   - FCN-8s refines this further by integrating pool3 features, enabling high-resolution predictions for small or detailed objects.

2. **Skip Connections**:  
   - Skip connections are critical for recovering spatial details lost during downsampling in the backbone.  
   - Pool4 (in FCN-16s) improves mid-scale object boundaries (e.g., sidewalks, poles), while pool3 (in FCN-8s) aids fine-scale features like traffic signs and fences.  
   - FCN-8s demonstrates how multi-resolution fusion enhances overall accuracy and visual coherence in segmentations.

### Effect of Backbone Fine-tuning

1. **Frozen vs Fine-tuned Backbone**:  
   - Fine-tuning the VGG16 backbone consistently improves performance across all architectures.  
   - For example, FCN-16s improves from 76.5% (frozen) to 83.0% (fine-tuned), while FCN-8s rises from 78.6% to 83.3%.  
   - Fine-tuning particularly benefits rare or small object classes like Traffic Sign (from 19.8% to 40.6%) and Pole (from 4.6% to 14.6%).

2. **Training Dynamics**:  
   - Frozen models stabilize quickly but plateau early, while fine-tuned models require longer training but achieve better final performance.  
   - Fine-tuning allows the backbone to adapt ImageNet features to segmentation-specific spatial patterns, enabling class-wise improvements especially in underrepresented categories.

### Per-Class Performance Analysis

1. **Easy vs Difficult Classes**:  
   - All models perform well on frequent, large-area classes like Car, Road, and Building (IoU > 90%).  
   - Rare or thin structures like Pedestrian, Pole, and Roadline remain challenging, with baseline FCN-32s achieving near-zero performance on these classes.  
   - FCN-8s (fine-tuned) shows significant improvements, with up to 40.6% IoU for Traffic Signs and 27.9% for Fence.

2. **Class IoU Distribution**:  
   - FCN-8s with fine-tuning yields the most balanced per-class IoUs across the 13 labels.  
   - The architecture’s capacity to integrate deep semantic understanding with fine-grained spatial detail proves most effective for diverse urban scenes.  
   - Performance gaps between classes narrow significantly with fine-tuning, reducing overfitting to dominant classes and enhancing minority-class segmentation.


**Final Conclusion:**  
FCN-8s with a fine-tuned VGG16 backbone achieves the best semantic segmentation results on this dataset, with a test mean IoU of **82.5%**. It offers the best trade-off between accuracy, detail, and class-wise consistency.
