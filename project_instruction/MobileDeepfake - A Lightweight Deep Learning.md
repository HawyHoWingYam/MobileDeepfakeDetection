# MobileDeepfake – Final System Design Overview

> Note: this English document was originally an early proposal.  
> It has been updated to describe the **final system** implemented in this repository and in the dissertation:  
> a two‑stage MobileNetV4 + EfficientNetV2 cascade with optional Stage‑3 meta‑model, exported to Android via ONNX Runtime.

# Application for Real-time Face-Swap Detection

## 1. Objectives of the Project

The primary objective of this project is to develop a **robust deepfake detection system**
specialized for face-swap manipulations and deploy it as a **mobile application**. The final
design is a **two-stage cascaded detector**:

- **Stage 1**: a lightweight MobileNetV4-based filter that quickly rejects easy real/fake cases.
- **Stage 2**: a higher-capacity EfficientNetV2-B3 expert that handles ambiguous samples.
- **(Optional) Stage 3**: a LightGBM meta-model used for offline analysis and research ablations, not part of the default mobile runtime.

The cascaded architecture focuses Stage 2 compute on hard examples while keeping Stage 1 fast enough for on-device use. Achieving reliable on-device detection is significant because deepfake technology, especially face swaps, has advanced to the point where manipulated media can be very convincing. While some state-of-the-art detectors can exceed **99% accuracy** on benchmark datasets (A Review of Deep Learning-based Approaches for Deepfake Content Detection), these solutions typically run on powerful hardware. By bringing deepfake detection to mobile devices, the project contributes to the accessibility of verification tools for the general public and the field of mobile AI.

In summary, the project’s goals are:

- Design a **cascaded binary classifier** that achieves very low false negative rate (FNR) under a bounded compute budget.
- Train and evaluate the cascade on multiple academic datasets (e.g., CelebDF‑v2, FaceForensics++, DFDC, DeeperForensics‑1.0), plus an in-the-wild benchmark.
- Export calibrated models and cascade thresholds to Android using **ONNX Runtime**, supporting fully on-device inference.

By accomplishing these objectives, the project contributes new insights into balancing
**detection accuracy and efficiency**, guiding future research in both deepfake detection and
mobile neural network deployment.

## 2. Background

**Deepfake Detection Techniques:** This section of the proposal provides a comprehensive
overview of deepfake detection, with an emphasis on face-swap scenarios. Deepfakes are

In practice, we instantiate these ideas as:

- **Deepfake Detection Model**: A two-stage cascade where Stage 1 is a MobileNetV4-Hybrid-Medium filter and Stage 2 is an EfficientNetV2-B3 expert. Both are fine-tuned for face-swap detection at 256×256 resolution and calibrated on a combined validation set.
- **Mobile Deployment**: Integration and optimization within an Android application using ONNX Runtime. The app runs the same Stage 1/2 models and calibrated thresholds as the desktop cascade, enabling privacy-preserving, fully on-device inference.
- **Innovation and Contribution**: A complete training → tuning → robustness → deployment pipeline that treats cascaded inference, calibration, and mobile export as a single system, rather than isolated components.

typically created using advanced generative models (such as **GANs** or **autoencoders** ) to
replace one person’s face with another’s in images or videos. Detecting such forgeries relies
on finding subtle inconsistencies or artifacts left by the manipulation process. A common
approach is to use Convolutional Neural Networks (CNNs) to automatically learn and identify
these artifacts in images. Prior works have explored various CNN architectures and even
temporal models (like LSTMs) to catch anomalies **frame-by-frame in videos**. For instance,
Rössler et al. (2019) evaluated multiple CNN-based detectors on face-swap videos and
found that an **XceptionNet CNN** (with depthwise separable convolutions) outperformed
other architectures in detecting face-swapped faces (Deepfakes: Face synthesis with GANs
and Autoencoders | AI Summer). Such techniques achieve high accuracy on test datasets,
but maintaining this performance in real-world conditions (with compression, noise, or novel
deepfake methods) remains challenging. Therefore, the background review will also discuss
alternative detection cues (e.g., physiological signal analysis, blinking patterns, or texture
inconsistencies) and why a CNN-based image analysis approach is suitable for a mobile
application focusing on face swaps

**Datasets for Deepfake Detection:** An effective deepfake detector requires a substantial
dataset of both real and fake (face-swapped) examples for training and evaluation. The
proposal will cover prominent datasets in this domain. A key dataset is **FaceForensics++** , a
large-scale benchmark containing manipulated videos. FaceForensics++ consists of **1000
original video sequences** (extracted from YouTube) that have been manipulated with four
automated face manipulation methods: **Deepfakes** , **Face2Face** , **FaceSwap** , and
**NeuralTextures** (FaceForensics++ Dataset | Papers With Code). It provides both high-
quality (less compressed) and low-quality (compressed) versions of videos, allowing
researchers to test detection algorithms under varied conditions. This dataset is highly
relevant as it includes face-swap forgeries, exactly the kind our project targets. We will detail
how the real and fake video frames from FaceForensics++ will be extracted and used to
train the model. Additionally, the overview will mention other datasets like the **Deepfake
Detection Challenge (DFDC)** dataset and **ForgeryNet** for context. The DFDC, for example,
is a large-scale public dataset with thousands of deepfake videos released to spur research
in detection. While our project will primarily use FaceForensics++ (focused on face swaps),
awareness of these other datasets is important for understanding the state of the field and
possibly for future generalization tests.

**CNN Models under Consideration:** The proposal will elaborate on the CNN architectures
being considered for detecting face swaps, with attention to their suitability for mobile
deployment. We will review models such as **XceptionNet, MobileNet, and EfficientNet-lite** ,
among others. XceptionNet is notable because of its success in deepfake detection research
(Deepfakes: Face synthesis with GANs and Autoencoders | AI Summer), achieving top
accuracy by leveraging depthwise separable convolutions to efficiently capture manipulation
artifacts. However, Xception is a relatively heavy model in terms of parameters and
computations. In contrast, MobileNet and EfficientNet-lite are designed to be lightweight and
faster on mobile devices while still performing well on image classification tasks. The
background section will discuss the trade-offs of these models: for example, MobileNet’s use


of depthwise separable convolutions drastically reduces computation, making it feasible for
real-time inference on a phone, though possibly at some cost to ultimate accuracy.
EfficientNet variants scale depth, width, and resolution in a balanced way; a smaller
EfficientNet (such as B0 or a tailored version) might offer a good accuracy-speed
compromise. We will also mention any specialized models from literature aimed at deepfake
detection, consider if elements from those can be incorporated. The relevance of each
model to mobile deployment will be analyzed in terms of model size, parameter count, and
known performance on devices.

**Dataset Augmentation and Evaluation Strategy:** To ensure the CNN generalizes well and
is robust to various input conditions, we will employ data augmentation techniques. The
background section will describe planned augmentations such as random horizontal flips,
rotations, slight blurring, color jittering, and compression artifacts injection. These
augmentations simulate real-world transformations (e.g., a deepfake image might be re-
compressed when shared on social media, or a video frame might be slightly motion-blurred)
and help the model not to overfit to a narrow data distribution. We will also outline the
dataset split and evaluation methodology. Likely, the dataset will be divided into training,
validation, and test sets, ensuring that **no overlap in source videos** occurs between
training and test to rigorously evaluate performance on unseen content. If multiple datasets
are used, cross-dataset evaluation will be mentioned – for instance, training on dataset and
testing on a different set to gauge generalization. The proposal content will emphasize
metrics used in evaluation, such as accuracy, precision, recall, F1-score, and ROC–AUC, or
some new evaluation metric, explaining why each is important for understanding the
detector’s effectiveness. We will also discuss evaluating the model’s robustness: for
example, how performance changes under lower video quality or adversarial perturbations,
since a mobile app may encounter uploads of varying quality.

In summary, the **Background** section of the thesis will demonstrate a thorough
understanding of deepfake detection research to date (techniques and challenges), the data
resources available for training detectors, and the deep learning models that are candidates
for our implementation. This builds a strong foundation for the choices made in our
methodology.

## 3. Methodology

Our methodology focuses on designing and training a deepfake detection model that is both
accurate and efficient enough for mobile deployment. We maintain the original structure of
data usage and model training, but **infuse state-of-the-art optimizations** at each stage to
produce a compact, high-performance model. The process can be summarized in several
key steps:

### Dataset Preparation and Usage

We will leverage a **large-scale deepfake video dataset** to train and evaluate the models.
For example, the **FaceForensics++** dataset (which contains 1,000 real videos and their


manipulated versions created with four different face-swap methods) could serve as a
primary training source (FaceForensics++: Learning to Detect Manipulated Facial Images).
This dataset provides a diverse set of fake videos generated by techniques such as
FaceSwap, DeepFakes, Face2Face, and NeuralTextures (FaceForensics++: Learning to
Detect Manipulated Facial Images), ensuring the model learns a wide variety of manipulation
artifacts. We will preprocess the videos by extracting frames (or short clips) and applying
data augmentations (e.g., random cropping, flipping, color jitter) to improve the model’s
robustness. The dataset will be split into training, validation, and test sets. **During training,
### Model Selection: Cascaded Architecture

In the final system we adopt a **cascaded architecture** instead of a single teacher–student pair:

- **Stage 1 (MobileNetV4-Hybrid-Medium)** acts as a fast, low‑compute filter that operates on 256×256 face crops and outputs a calibrated fake probability.
- **Stage 2 (EfficientNetV2-B3)** is a higher-capacity expert that processes only those samples whose Stage‑1 scores fall in an “uncertain band” defined by low/high thresholds.
- **Stage 3 (LightGBM meta-model, optional)** can fuse features and logits from Stage 2 (and optional research experts) for offline analysis, but is not part of the default mobile runtime.

This design directly mirrors the dissertation: Stage 1 and Stage 2 are each trained in a standard supervised fashion on combined multi‑dataset manifests; Stage 3 is used only when explicitly enabled for research ablations.

### Training Procedure (Final System)

Training proceeds in stages:

1. **Stage 1 training and calibration**
   - Train MobileNetV4 on balanced multi‑dataset manifests with a standard binary classification loss.
   - Apply post‑hoc temperature scaling on a held‑out validation split to obtain a calibrated fake probability for cascade threshold tuning.

2. **Stage 2 training (EfficientNetV2-B3)**
   - Train EfficientNetV2-B3 on the same combined manifest with stronger augmentation (e.g., RandAugment, Mixup, CutMix) and a focal-style loss.
   - Optionally construct a “hard subset” based on Stage‑1 scores for ablation runs with hard example mining; the default configuration uses uniform sampling.

3. **Stage 3 meta-model (optional)**
   - Build K‑fold meta-datasets from out‑of‑fold Stage‑2 embeddings and logits.
   - Fit a LightGBM classifier to probe whether feature-level fusion improves over the best single expert. In practice, the meta‑model offers limited gains and remains an offline research component.

4. **Cascade threshold tuning**
   - Sweep low/high thresholds on validation data to trade off FNR against the Stage‑2 escalation rate.
   - Select a “safety‑first” operating point (very low FNR with low Stage‑2 usage) and export these thresholds for deployment.

### Model Compression via Pruning and Fine-Tuning

After knowledge distillation, we will further compress the student model using **network
pruning**. Pruning aims to remove unnecessary or low-impact weights/neurons from the
model, thereby reducing its size and speed requirements without significantly affecting
accuracy (A Comparative Analysis of Compression and Transfer Learning Techniques in
DeepFake Detection Models). According to prior studies, **pruning is one of the most
widely-used techniques to eliminate redundant parameters in neural networks** (A
Comparative Analysis of Compression and Transfer Learning Techniques in DeepFake
Detection Models). Our pruning strategy will be as follows:

```
OpenReview). The teacher’s output acts as a softened target that contains richer
information (it may encode how strongly the teacher believes a video is fake vs real,
which frames are tricky, etc.). By mimicking the teacher’s outputs, the student can learn
the subtle patterns and decision boundaries the teacher discovered, even if the student
model is much smaller (A Comparative Analysis of Compression and Transfer Learning
Techniques in DeepFake Detection Models). We will use a temperature scaling strategy
for the soft targets (as per Hinton’s distillation approach) to ensure the teacher’s
knowledge is effectively transferred. During distillation training, the teacher’s weights
remain fixed; only the student is updated. This student–teacher learning paradigm has
shown benefits like improved efficiency and preservation of performance on edge-sized
models (A Comparative Analysis of Compression and Transfer Learning Techniques in
DeepFake Detection Models) (A Comparative Analysis of Compression and Transfer
Learning Techniques in DeepFake Detection Models). Our training will iterate until the
student’s performance on validation data nearly matches the teacher’s. We expect that
with a well-tuned distillation process, the student model will approach the teacher’s
accuracy while having far fewer parameters (A Comparative Analysis of Compression
and Transfer Learning Techniques in DeepFake Detection Models). We will document
how close the student gets (e.g., perhaps within a few percentage points of accuracy of
the teacher). If the gap is large, we might adjust hyperparameters (like loss weighting or
training epochs) or consider intermediate feature distillation: using not just outputs but
also intermediate feature maps from the teacher to guide the student, as suggested in
some research (A novel model compression method based on joint distillation for
deepfake video detection | OpenReview), though that adds complexity. The primary
outcome of this stage is a trained lightweight student model that performs almost as
well as the heavy teacher model on deepfake detection.
```
```
We will start with the distilled student model (which is already smaller than the teacher)
and evaluate its weight magnitudes. Using a criterion such as L1-norm magnitude , we
will identify the least important weights or filters in each layer (A Comparative Analysis of
Compression and Transfer Learning Techniques in DeepFake Detection Models).
Specifically, unstructured pruning can be applied, where individual weights below a
certain threshold are set to zero (or removed) (A Comparative Analysis of Compression
and Transfer Learning Techniques in DeepFake Detection Models). To maintain a
```

### Quantization and Efficient Mobile Deployment

The final step is to prepare the optimized cascade for deployment on mobile devices, which
involves **model quantization** and integration into a mobile app framework. Quantization
compresses the models by reducing the numerical precision of the weights (and possibly
activations), typically from 32-bit floating point to 8-bit integers. This can **shrink memory and
storage requirements by ~75%** and speed up inference using integer arithmetic (What Is
int8 Quantization and Why Is It Popular for Deep Neural Networks? - MATLAB & Simulink)
while keeping accuracy within a small margin.

In the final system we:

- Apply **post-training dynamic quantization** to linear layers in the TorchScript exports of Stage 1 and Stage 2;
- Optionally explore pruning or quantization-aware training in research variants when further efficiency is needed.

The quantized models are then exported to ONNX and bundled together with thresholds and calibration parameters in a lightweight metadata JSON. On Android, **ONNX Runtime** loads these models once at app startup and executes the two-stage cascade on-device.

**Mobile Integration:** Once quantized and exported, we integrate the cascade into a mobile application prototype. The app takes images or frames from the device (e.g., gallery or video) and runs the two-stage cascade on each face crop. Because the models are carefully optimized and partially quantized, the system achieves practical per-image latency on representative smartphones while keeping storage and memory usage modest. We measure inference speed, memory footprint, and energy characteristics on a target device to ensure that the app can process media on-device without noticeable lag or excessive battery drain.

**Deployment Strategy:** The deployment will emphasize **on-device processing for privacy
and latency**. By not sending data to a server, the deepfake detection happens locally,
preserving user privacy (sensitive videos never leave the device) and working even without
network connectivity. We will include in the proposal a plan for **field testing** : having a demo
app that users or testers can use to identify deepfakes in videos or live camera feed. The
output could be a confidence score or alert if a deepfake is detected. We also plan to include
a fallback or update mechanism – for example, if a new kind of deepfake appears that the
model struggles with, we can retrain the teacher on new data and repeat the distillation and
compression pipeline, then update the mobile model. This ensures the solution stays
resilient as deepfake technology evolves.

Throughout deployment, we will keep an eye on the **consistency between the
development and production environments**. The dataset and model selection decisions
made earlier (e.g., using a CNN architecture) are favorable for mobile, as CNN operations
(convolutions) are well-optimized on mobile hardware. By quantizing and pruning, we ensure
those operations are minimal and fast. Our approach is informed by research that highlights
the effectiveness of combining these optimization techniques for edge deployment (A


Comparative Analysis of Compression and Transfer Learning Techniques in DeepFake
Detection Models). In fact, **hybrid compression methods** (using distillation, pruning, _and_
quantization together) are known to yield models that are far more efficient while retaining
high accuracy (A Comparative Analysis of Compression and Transfer Learning Techniques
in DeepFake Detection Models). By integrating all three, our deepfake detector is expected
to be **orders of magnitude lighter** than a naive solution, yet nearly as accurate. For
example, if the teacher model was 200 MB in size, our final compressed model might end up
only a few megabytes, making it feasible to include in a mobile app download and to run
smoothly on device.

### Evaluation Metrics and Validation

To ensure that the project goals are met, we will rigorously evaluate the model at each stage
of optimization. Key metrics include: **classification accuracy** (or AUC) on the test set of
deepfake videos, to measure detection performance; **model size (storage)** in megabytes
and number of parameters; **inference latency** (milliseconds per frame) on a target mobile
device; and **energy/memory usage** on device if possible. We will compare the student
model’s accuracy to the teacher’s accuracy to quantify the knowledge distillation success.
We’ll also compare the pruned/quantized model’s accuracy to the uncompressed student’s
accuracy to verify that compression did not significantly hurt performance. Our target is to
keep any accuracy drop within a small margin (e.g., < 2-3%). If the drop is larger, we will
revisit the training or compression parameters. We will document these results in the
proposal’s evaluation section, showing the trade-offs clearly. Additionally, we will test the
final mobile app with a variety of videos (both from the dataset and real-world samples not
seen in training) to ensure the system works on **practical inputs**. The final deliverable will
include both the research findings (accuracy vs. efficiency numbers) and the demonstration
of a working mobile deepfake detection prototype.

## Scheduled Program of Work

A clear timeline is outlined for completing the project in phases, ensuring all components—
from research to implementation and evaluation—are finished in a logical sequence. The
schedule for the work is as follows:

```
Month 1-2: Literature Review and Proposal Refinement – Conduct an in-depth
literature review on deepfake generation and detection techniques, with special focus on
face-swap detection methods and prior mobile AI implementations. During this phase,
gather relevant research papers, survey existing deepfake detection tools, and
familiarize yourself with practical deployment stacks (PyTorch, TorchScript, ONNX Runtime).
Simultaneously, refine the thesis proposal (this document) based on initial findings and
feedback, clearly defining the project scope and plan.
Month 2-3: Dataset Acquisition and Preparation – Obtain the FaceForensics++
dataset (and any supplementary datasets such as DFDC or others if needed). Set up the
data processing pipeline: extract frames from videos if working with still-image detection,
```

or prepare video clips if doing video-based detection. Perform data cleaning (ensure all
files are in usable format, and pair each fake with its corresponding real counterpart if
needed). Apply data augmentation techniques to expand the training set diversity. By the
end of this phase, have a ready-to-use training dataset split into training, validation, and
test sets.
**Month 3-4: Model Training and Development** – Begin with implementing the chosen
CNN model architectures in a training framework (PyTorch/TensorFlow). Train initial
models on the prepared dataset, tuning hyperparameters (learning rate, batch size,
epochs) to achieve good validation performance. If experimenting with multiple models
(e.g., Xception vs MobileNet), train each and compare results. This phase may involve
iterative experimentation: adjusting the model or training strategy (such as using class
weighting or focal loss if the dataset is unbalanced, etc.) and retraining. Aim to obtain a
model that reaches the target accuracy on the validation set. Periodically evaluate on the
hold-out test set to estimate real performance, but the bulk of test evaluation is left for
later to avoid bias.

**Month 5: Model Optimization and Conversion** – Once satisfactory Stage 1 and Stage 2 models are trained,
optimize them for mobile deployment. This includes exporting the models to a portable
format (TorchScript and ONNX) and validating that post-training quantization preserves accuracy.
Export the cascade to ONNX and verify on desktop (or emulator) that ONNX Runtime produces
the same outputs as the original PyTorch models. Iterate on this process if any issues arise
(for example, if an operator is unsupported, adjust the model or export settings and re-test).
By the end of this month, have the quantized ONNX models and a small cascade configuration
JSON ready for integration into the Android app.

**Month 6: Android Application Development** – Develop the Android app that will host
the deepfake detection functionality. Set up the Android Studio project, add ONNX Runtime
for Android as the inference engine, and implement a Kotlin-based cascade engine that
loads the exported ONNX models and configuration. Design and implement the user interface,
ensuring it is intuitive. Integrate image selection or (later) camera capture features for users
to input media. Connect the UI with the detection logic: when the user submits an image,
run the Stage 1 / Stage 2 cascade and return the result to the UI, including label, confidence,
stage used, and timing. Conduct preliminary testing with sample images to verify end-to-end
functionality (correct model loading, inference, and result display).

**Month 7: Testing and Evaluation** – Rigorously test the entire system. This involves two
aspects: (a) **Accuracy Testing:** Run the app (or the underlying model) on the reserved
test dataset of deepfakes and real images to compute the final accuracy, precision,
recall, F1, etc. Compare these metrics to what was observed during offline testing to
ensure the mobile integration did not introduce any issues. (b) **Performance and
Usability Testing:** Deploy the app on at least one or two physical Android devices with
different hardware capabilities (for example, a high-end phone vs. a mid-range phone).
Measure inference times for images and short videos, check memory usage, and
monitor for any crashes or slowdowns. Gather feedback on the usability of the UI (e.g.,
from a few trial users or colleagues). If any performance bottlenecks are identified (such


This schedule provides a structured plan to ensure that the project stays on track. Each
phase builds upon the previous one: the literature review informs the model choice, which
once trained is optimized and embedded into an application, which is then tested and
analyzed. Regular milestones (such as a functional model by Month 4 and a working app by
Month 6) ensure that any issues can be identified early and addressed. By following this
timeline, the project will systematically achieve its objectives and culminate in a successful
thesis and a demonstrable mobile deepfake detection application.

```
as inference being too slow), consider last-minute optimizations (like reducing input
resolution or further quantizing the model) and test again. Ensure that the detection
results are reliable and the app is stable.
Month 8: Documentation and Thesis Writing – Compile all results, analyses, and
insights gained throughout the project and begin writing the thesis document. This will
include documenting the research background, the methodology (model design, training
process, integration steps), and the outcomes (experimental results and the final app’s
performance). In this phase, create necessary figures (such as model architecture
diagrams, training loss/accuracy plots, and perhaps screenshots of the app in action) to
include in the thesis. Write and refine each chapter, maintaining an academic tone and
proper citations for any literature references. Aim to finalize the thesis by the end of
Month 8, leaving additional time for proof-reading and revisions as needed before
submission.
```
