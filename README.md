# Lung Segmentation from X-ray

## Introduction

Lung segmentation is a crucial step in the analysis of chest X-ray images, as it allows for the isolation of lung regions from surrounding anatomical structures. This process is essential for accurate diagnosis and monitoring of various lung diseases, such as pneumonia, lung cancer, and tuberculosis. By automating lung segmentation, we can expedite the analysis process, reducing the workload for medical professionals and enhancing patient care.


## Model Architecture

Our lung segmentation model employs a unique and powerful architecture that combines the strengths of UNet, ResNet, and semantic-guided attention to achieve high accuracy and efficiency. 

### UNet with ResNet Backbone

UNet is a popular architecture for medical image segmentation tasks, owing to its symmetric encoder-decoder structure and skip connections that facilitate precise localization. We have enhanced the UNet architecture by incorporating a ResNet backbone, which improves feature extraction capabilities and enables the model to learn complex patterns in the X-ray images more effectively.

### Semantic-Guided Attention

To further refine our model, we integrated a semantic-guided attention mechanism into the architecture. This attention module helps the model to focus on critical regions in the image by assigning higher importance to lung areas during the segmentation process. By incorporating this attention mechanism, our model can better handle challenging cases, such as those with overlapping structures or low contrast.

Together, these components form a powerful architecture that is capable of delivering accurate and efficient lung segmentation results.



## Dataset:

The Montgomery County Tuberculosis Control Program dataset is a collection of chest X-ray images provided by the National Library of Medicine (NLM), National Institutes of Health (NIH), and the Department of Health and Human Services, USA. This dataset is comprised of images from the tuberculosis control program in Montgomery County, Maryland, and is primarily used for researching lung diseases, particularly tuberculosis.

The dataset consists of both normal and abnormal chest X-ray images, with associated metadata and annotations. These images are often used to train and evaluate machine learning models, such as lung segmentation algorithms, for the detection and diagnosis of various lung diseases. It is a valuable resource for researchers and developers working on medical image processing and analysis in the field of radiology.


##Result:

Dice score achieved: 89%

![result image](https://github.com/mmuttir/Lung-Segmentation-in-Xray-Images/blob/main/predicted.png)
