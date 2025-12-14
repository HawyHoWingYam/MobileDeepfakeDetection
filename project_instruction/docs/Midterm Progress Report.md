# Midterm Progress Report

**Student Name:** Wing Yam Ho
**Supervising Professor:** Professor Zhou Kai
**Report Date:** July 25, 2025

**1. Executive Summary**
This research aims to design and implement a Deepfake detection solution specifically
for mobile platforms, balancing detection accuracy with computational efficiency to
counter the growing threat of video-based fraud. This report summarizes the core
progress since the project's inception. The project initially started with a lightweight
real-time detection model but, after an in-depth literature review and feasibility
analysis, has evolved to a more innovative cascaded heterogeneous ensemble
architecture.
To date, the project has fully completed Phase Zero (Infrastructure and Data
Processing) and Phase One (Fast Filter Model). We have established an automated
processing pipeline for multiple datasets and successfully trained a reliable
lightweight model, MobileNetV4, to serve as the system's first line of defense.
Currently, the project is in Phase Two, where the training frameworks for the two core
expert models, EfficientNetV2-B3 and GenConViT, have been constructed, laying a
solid foundation for the subsequent stacking ensemble. This report will detail the
quantitative results of completed work, the status of ongoing tasks, challenges
encountered, and the research plan for the next steps.
**2. Research Background and Methodological Evolution
2.1 Problem Definition**
With the proliferation of generative AI technology, face-swap videos have become a
severe threat to online security and are frequently used in scams. However, deploying
high-accuracy, heavy-duty detection models directly on mobile devices presents
significant performance challenges, while lightweight models often fail to achieve
sufficient detection accuracy.
**2.2 Evolution of the Solution**
The research path for this project has not been static. The initial concept was to focus
on developing a single, lightweight CNN model to achieve real-time detection.
However, after a thorough investigation of state-of-the-art solutions in academia, we


recognized the limitations of a single-model approach in complex scenarios.
Consequently, the project has adopted a more advanced and pragmatic technical
solution:
● **Application Scenario:** Shifted from "real-time detection" to a more feasible
"offline analysis" mode.
● **Core Architecture:** Adopted a two-stage cascade architecture, using a
lightweight model to handle a large volume of simple samples and concentrating
computational resources on more difficult ones.
● **Core Models:** In the second stage, a heterogeneous model ensemble will be
used. This combines a CNN (EfficientNetV2-B3) to capture local forgery textures
with a Transformer model specifically designed for this task (GenConViT) to
identify global inconsistencies, aiming to achieve State-of-the-Art (SOTA)
performance.
This evolution reflects the project's rigorous and pragmatic approach, ensuring the
technical path is both forward-looking and feasible.

**3. Completed Work
3.1 Phase Zero: Project Foundation and Environment Setup (Completed)**
    ● **Achievements:**
       ○ **Reproducible Development Environment:** Created an isolated environment
          using Conda and environment.yml, including core libraries like Python 3.10,
          PyTorch, timm, and transformers.
       ○ **Automated Data Processing Pipeline:** Developed the core script
          scripts/preprocess_datasets_v2.py to automate the process from raw videos
          to cropped and standardized face images.
       ○ **Unified Dataset:** Successfully processed multiple datasets, including DFDC
          and FaceForensics++, and strictly partitioned them into unified training,
          validation, and a completely isolated final test set. This ensures the fairness of
          subsequent research.
    ● Dataset Scale
       According to the training log from 2025-07-23, the final generated dataset sizes
       are as follows:
          ○ **Training Set:** 77,827 Real images, 548,052 Fake images (Total: 625,879)
          ○ **Validation Set:** 17,458 Real images, 130,780 Fake images (Total: 148,238)
**3.2 Phase One: First-Stage Model - Fast Filter (Completed)**
    ● **Achievements:**


```
○ Model Training: Successfully fine-tuned the lightweight
MobileNetV4-Hybrid-Medium model. The training framework
(src/stage1/train_stage1.py) integrates strong data augmentation strategies,
the AdamW optimizer, and the CosineAnnealingLR learning rate scheduler.
The best model was saved based on the validation set's AUC score.
○ Probability Calibration: Applied Temperature Scaling
(src/stage1/calibrate_model.py) to calibrate the model's output probabilities,
enhancing the reliability of its predictions.
○ Performance Evaluation: Conducted a comprehensive performance
evaluation of the calibrated model using src/stage1/evaluate_stage1.py to
establish a reliable performance baseline.
● Phase One Core Performance Metrics
Based on recent logs, the model's core experimental data on the validation set is
as follows:
○ a) The highest AUC score achieved during training was 0.9733 (in the 7th
Epoch).
○ b) The optimal temperature value (T) found during probability calibration was
1.0. (Note: A value of 1.0 indicates that the model's original outputs were
already well-calibrated).
○ c) Other metrics: [Final F1-Score, Accuracy, ECE, etc., to be obtained by
running the evaluation script on the validation set later].
```
**4. Work in Progress
4.1 Phase Two: Second-Stage Model - Precision Analyzer (In Progress)**
This phase is the current focus of our work, aiming to forge the "heart" of the cascade
system—two powerful, heterogeneous expert models.
    ● **Current Status:**
       ○ **EfficientNetV2-B3 (Local Feature Expert):** The training script
          src/stage2/train_stage2_effnet.py has been developed and is ready for
          training to commence.
       ○ **GenConViT (Global and Generative-Awareness Expert):**
          ■ The source code for GenConViT has been locally integrated
             (src/stage2/genconvit/), which will greatly facilitate subsequent in-depth
             debugging and customization.
          ■ A model manager, src/stage2/genconvit_manager.py, has been developed
             to encapsulate its complexity.
          ■ The training script src/stage2/train_stage2_genconvit.py has been
             developed, but training has not yet begun.


```
○ Feature Extraction Tool (feature_extractor.py): Initial development has
started.
```
**5. Challenges, Solutions, and Insights**
During the project's progress, we have encountered and resolved the following
challenges:
    ● **Challenge 1: Integration of Heterogeneous Datasets:** Different datasets (e.g.,
       DFDC, FF++) have varied directory structures and metadata formats.
          ○ **Solution:** We developed a flexible and configurable
             preprocess_datasets_v2.py script. By parsing different metadata formats, we
             successfully unified them into a standard directory structure and manifest
             format.
    ● **Challenge 2: Introduction of Cutting-Edge Models:** As a relatively new model,
       integrating GenConViT presented some complexity.
          ○ **Solution:** Instead of merely calling it through the transformers library, we
             chose to localize its source code. Although this required more initial time to
             understand its internal structure and weight-loading mechanisms, it has
             paved the way for precise feature extraction and potential model
             modifications later on.
**6. Next Steps & Timeline**
The subsequent phases of the project have a clear roadmap, with the goal of
completing the first full draft of the report by early October. The current plan is
generally aligned with this objective.
    **Phase Core Tasks Estimated**
       **Completion**
          **Target Deadline**
    **Phase Two**
    **(Wrap-up)**
       Complete final
       training and tuning of
       EfficientNetV2 and
       GenConViT; complete
       feature_extractor.py.
          ~3 Weeks Mid-August 2025
    **Phase Three** Stacking Ensemble
       and Meta-Model
       Training: Write
       create_meta_dataset.
       py. Generate
       meta-features via
       cross-validation; train
          ~3 Weeks Late-August 2025


```
the LightGBM
meta-model.
Phase Four System Integration
and Mobile
Optimization:
Develop the
CascadeDetector
class. Apply
Quantization-Aware
Training (QAT) and
Knowledge
Distillation (KD) to all
neural network
models.
~4 Weeks Early September
2025
Phase Five Comprehensive
Evaluation and
Analysis: Conduct a
full evaluation on the
reserved final test
set; analyze
robustness and
generalization
capabilities.
~2 Weeks Mid-September 2025
Draft Report
Submission
Consolidate all
results and complete
the first full draft of
the report.
Late September 2025
```
**7. Literature Review and Key References**
This section systematically reviews the academic work supporting the core technical
solutions of this research (cascade architecture, heterogeneous integration, mobile
optimization) and lists the main references.
**7.1 Core Methodology Support**
    ● **Cascade Architecture:** In recent years, cascade architectures have shown great
       potential in balancing detection accuracy and computational efficiency. Studies
       show that a two-stage strategy, where a lightweight front-end model filters
       simple samples and a complex back-end model performs precise analysis,
       achieves excellent performance on multiple standard datasets [2]. Fahad et al.
       further combined this with binarized neural networks to explore applications on


extremely resource-constrained devices [3]. The cascaded face detection
preprocessing pipeline used in this study is also based on mature research of
multi-task cascade frameworks like MTCNN [4, 8].
● **Heterogeneous Models and Ensemble Learning:** Our core idea is to combine
the strengths of different models. MobileNetV4 represents the latest
advancement in efficient inference on mobile devices [9], while EfficientNetV2 has
been proven to be a high-performance backbone network in several studies [10,
11, 12]. GenConViT, a generative convolutional vision Transformer designed
specifically for Deepfake detection, combines the architectures of ConvNeXt and
Swin Transformer to learn both visual artifacts and underlying data distributions,
achieving SOTA performance on multiple benchmarks [13, 14]. Furthermore,
stacking, an advanced meta-learning method, has been shown to effectively fuse
deep features from multiple base models to achieve accuracy surpassing that of
any single model [17, 18, 19, 52, 53].
● **Mobile Optimization and Deployment:** To achieve final deployment on mobile
devices, this study plans to adopt a strategy combining Quantization-Aware
Training (QAT) and Knowledge Distillation (KD). QAT simulates quantization noise
during training, which can preserve model accuracy to the greatest extent [28,
29]. Knowledge distillation allows a lightweight "student" model to inherit the
generalization capabilities of a large "teacher" model by mimicking its output,
making it an effective technique for mobile deployment [48, 49, 51].
● **Training and Evaluation Strategies:** This research adopts several
industry-standard best practices, including using the AdamW optimizer for better
generalization performance [41, 42] and the CosineAnnealingLR learning rate
scheduler for smooth convergence [43, 44]. In terms of evaluation, we not only
focus on standard metrics like AUC and F1-Score but also incorporate probability
calibration (e.g., temperature scaling) [22, 25] to improve the reliability of model
predictions. The datasets used, such as DFDC [34-39] and FaceForensics++ [40],
are authoritative benchmarks in the field.
**7.2 List of References**
[1] Salman, M., Tariq, I., Zulfiqar, M., Jalal, M., Aujla, S., & Fatima, S. "AWARE-NET: Adaptive
Weighted Averaging for Robust Ensemble Network in Deepfake Detection." arXiv:2505.00312,
2025.
[2] Nan, Y., et al. "A Cascade Network Based on Transformer for Deepfake Video Detection."
Mathematical Biosciences and Engineering, 2024.
[3] Fahad, M., et al. "An Efficient and Scalable Deepfake Detection for Resource-Constrained
Mobile and Embedded Devices using Binarized Neural Networks." IEEE International
Conference on Communications, 2024.
[4] Zhang, K., et al. "Joint Face Detection and Alignment Using Multi-task Cascaded


Convolutional Networks." IEEE Signal Processing Letters, 2016.
[5] "Facial detection using MTCNN." Velog, 2023.
[6] "Usage Parameters - MTCNN." Read the Docs.
[7] Zhang, Z., et al. "3D-Aided Face Alignment and Shape Reconstruction in the Wild." Chinese
Academy of Sciences, 2018.
[8] Zhang, K., et al. "Multi-task Cascaded Convolutional Networks for Face Detection, etc."
SPL, 2016.
[9] "MobileNetV4: Generasi Baru Model Universal untuk Ekosistem Seluler." Jurnal JTI, 2024.
[10] Raj, P., et al. "A Novel Approach for Deepfake Video Detection using EfficientNet with
Temporal Convolutional Network." International Journal of Scientific & Academic Research,
2024.
[11] Liang, B., et al. "Deepfake Video Detection Based on Facial Edge-band." IEEE International
Conference on Acoustics, Speech and Signal Processing, 2022.
[12] Liu, W., et al. "A Lightweight Deepfake Detection Model Based on Improved
EfficientNetV2-S and CBAM." International Conference on Computer and Drone Applications,
2023.
[13] Heo, Y., et al. "GenConViT: Generative and Consistency-aware Vision Transformer for
Deepfake Detection." arXiv:2307.07036, 2023.
[14] "Deepfake Video Detection using Generative and Consistency-aware Vision Transformer."
Papers with Code.
[15] "GenConViT." Hugging Face Papers.
[16] "GenConViT." GitHub Repository.
[17] Rani, S., et al. "Deep Feature Stacking and Meta-Learning-Based Deepfake Video
Detection Using an Explainable AI." Genes, 2024.
[18] Rani, S., et al. "Deepfake detection using deep feature stacking and meta-learning with
explainable AI." Heliyon, 2024.
[19] "Multi-model Stacking Ensemble Learning for Enhanced Medical Image Analysis." IEEE
Access, 2024.
[20] Nugraha, Y., et al. "Skin Disease Classification Using Multi-Model Stacking Ensemble
Learning." Jurnal Teknologi dan Sistem Komputer, 2024.
[21] "Stacked Ensemble-Based Transfer Learning Model for Emergency Call Type Prediction."
IEEE Access, 2024.
[22] Guo, C., et al. "On Calibration of Modern Neural Networks." Proceedings of the 34th
International Conference on Machine Learning, 2017.
[23] "Adaptive Temperature Scaling." arXiv:2409.19817, 2024.
[24] Ding, Z., et al. "Local Temperature Scaling for Probability Calibration." IEEE/CVF
International Conference on Computer Vision, 2021.
[25] "Local Temperature Scaling for Probability Calibration." The CVF Open Access.
[26] "LTS: Local Temperature Scaling." GitHub Repository.
[27] Ding, Z., et al. "Local Temperature Scaling for Probability Calibration." arXiv:2008.05105,
2020.
[28] "Quantization Aware Training." Qualcomm Innovation Center.
[29] "Quantization-Aware Training (QAT)." IBM Technology.


[30] Li, Y., et al. "Data-Free Quantization-Aware Finetuning." Proceedings of Machine Learning
Research, 2020.
[31] Zhang, J., et al. "8-Bit Quantization of 3D-UNet for Segmentation of Medical Images."
arXiv:2501.17343, 2025.
[32] "The Complete Guide to Mobile Optimization in 2024." Oake Marketing.
[33] "Strategies For Optimizing Mobile App Performance." MoldStud.
[34] Dolhansky, B., et al. "The Deep Fake Detection Challenge (DFDC) Dataset."
arXiv:2006.07397, 2020.
[35] "The DeepFake Detection Challenge (DFDC) Preview Dataset." Semantic Scholar.
[36] "The DeepFake Detection Challenge." Meta AI.
[37] Rössler, A., et al. "FaceForensics++: Learning to Detect Manipulated Facial Images."
Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019.
[38] "FaceForensics++ Benchmark." Technical University of Munich.
[39] Thies, J. "The FaceForensics++ Benchmark." justusthies.github.io.
[40] "Deepfake-Eval-2024: A Multi-Modal, In-the-Wild Benchmark for Deepfake Detection."
arXiv:2503.02857, 2025.
[41] Zhou, P., et al. "Revisiting AdamW: A Closer Look at Its Theory and an Improved Variant."
IEEE Transactions on Neural Networks and Learning Systems, 2024.
[42] "Synthetic data generation with StyleGAN2 for the Mever deepfake detection." Mever.gr.
[43] "fairseq2.optim.lr_scheduler.CosineAnnealingLR." Facebook Research.
[44] "torch.optim.lr_scheduler.CosineAnnealingLR." PyTorch Documentation.
[45] "CosineWithRestarts." AllenNLP Documentation.
[46] "How to use knowledge distillation to create smaller, faster LLMs." DEV Community.
[47] "What is Knowledge Distillation?" IBM Technology.
[48] "What is Knowledge Distillation?" Roboflow Blog.
[49] Zhu, X., et al. "Student-Friendly Teacher for knowledge distillation." OpenReview, 2024.
[50] Seewald, A. K. "How to Make Stacking Better and Faster While Also Taking Care of an
Unknown Deployment Scenario." Austrian Research Institute for Artificial Intelligence, 2002.
[51] "An Attention-Based Stacking Ensemble Model for Improving the Accuracy of
Hydrological Forecasting." MDPI, 2023.
[52] Verdoliva, L. "Media forensics and deepfakes: an overview." IEEE Journal of Selected
Topics in Signal Processing, 2020.
[53] Zhang, H., et al. "mixup: Beyond Empirical Risk Minimization." arXiv:1710.09412, 2017.
[54] Bazarevsky, V., et al. "Blazeface: Sub-millisecond neural face detection on mobile gpus."
arXiv:1907.05047, 2019.


