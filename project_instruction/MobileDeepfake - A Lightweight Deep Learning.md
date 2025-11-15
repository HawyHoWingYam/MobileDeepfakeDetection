# MobileDeepfake - A Lightweight Deep Learning

# Application for Real-time Face-Swap Detection

## 1. Objectives of the Project

The primary objective of this project is to develop a **robust deepfake detection model**
specialized for face-swap manipulations and deploy it as a **mobile application**. This
involves using a **light-weight convolutional neural network (CNN)** capable of
distinguishing face-swapped images or video frames from genuine ones with high accuracy,
and optimizing this model to run efficiently on Android devices. Achieving reliable on-device
detection is significant because deepfake technology, especially face swaps, has advanced
to the point where manipulated media can be very convincing. While some state-of-the-art
detectors can exceed **99% accuracy** on benchmark datasets (A Review of Deep Learning-
based Approaches for Deepfake Content Detection), these solutions typically run on
powerful hardware. By bringing deepfake detection to mobile devices, the project contributes
to the accessibility of verification tools for the general public and the field of mobile AI.

In summary, the project’s goals are:

By accomplishing these objectives, the project will contribute new insights into balancing
**detection accuracy and efficiency** , guiding future research in both deepfake detection and
mobile neural network deployment.

## 2. Background

**Deepfake Detection Techniques:** This section of the proposal provides a comprehensive
overview of deepfake detection, with an emphasis on face-swap scenarios. Deepfakes are

```
Deepfake Detection Model : Develop a light-weight CNN-based model that can
accurately detect face-swap deepfakes, improving techniques in the domain of image
forensics and fake media detection. This model will be trained and fine-tuned specifically
to recognize artifacts from face swapping.
Mobile Deployment : Integrate and optimize this model within an Android application,
demonstrating on-device inference of deepfake detection. The significance lies in
contributing a practical tool that operates in real-time on consumer devices, which is a
step forward for mobile AI applications in computer vision.
Innovation and Contribution : Advance the deepfake detection field by focusing on
resource-constrained environments. The project will show how a high-performing
deepfake detector can be compressed and accelerated for mobile use, using
frameworks like NCNN, thus bridging the gap between cutting-edge deepfake detection
research and user-facing mobile technology. This not only aids in combating
misinformation but also highlights methods for deploying AI models in portable contexts.
```

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
the large “teacher” model will be trained on this dataset first** , learning to distinguish real
vs fake patterns with high accuracy. Subsequently, the **“student” model (a lightweight
network)** will be trained on the same data using knowledge distillation (detailed below),
effectively **using the dataset twice** : first to fit the teacher, then to transfer knowledge to the
student. We will ensure that the student model sees the full diversity of training data (and
any available augmentation) so that despite its smaller size, it generalizes well. For
evaluation, we will test both models on a held-out set of videos and possibly on additional
benchmark deepfake datasets (such as DFDC or others) to verify that the compressed
model maintains high accuracy across different data sources.

### Model Selection (Teacher and Student Architecture)

```
Teacher Model: As a baseline for top accuracy, we will start with a proven deepfake
detection architecture. One option is the Xception network , which has demonstrated
high performance in deepfake image classification tasks (it was used as a top performer
in the original FaceForensics++ benchmark) (FaceForensics++: Learning to Detect
Manipulated Facial Images). The Xception model (or a similar CNN) is deep and high-
capacity, which makes it ideal to serve as a teacher that learns subtle forgery cues in the
training data. However, such a model is too heavy for mobile deployment – Xception has
over 22 million parameters – so it will be used only for offline training on a powerful
machine. Another candidate could be a modern vision transformer or an ensemble of
CNNs for even higher accuracy, but to keep the process feasible and given the need
for compression, a CNN-based teacher is preferred. CNNs benefit from well-
established compression techniques (pruning, quantization, etc.) that have been refined
for edge devices (A Comparative Analysis of Compression and Transfer Learning
Techniques in DeepFake Detection Models), whereas transformer models are still being
optimized for such constraints (A Comparative Analysis of Compression and Transfer
Learning Techniques in DeepFake Detection Models). Thus, our approach prioritizes a
CNN teacher model to ensure easier compression and efficient knowledge transfer to
the student.
Student Model (Efficient Architecture): For the mobile-friendly student network, we will
select an efficient CNN architecture known to run well on limited hardware. A strong
candidate is MobileNet-V3 , a lightweight convolutional network with roughly 1.5 million
parameters , specifically designed for mobile applications ([2205.00211] DefakeHop++:
An Enhanced Lightweight Deepfake Detector). MobileNet-V3 uses depthwise separable
convolutions and other optimizations to drastically reduce computation while still
```

### Training Procedure and Knowledge Distillation

The training will be carried out in two stages: first training the teacher, then training the
student via knowledge distillation.

```
performing well on vision tasks. By using MobileNet (or a similar efficient architecture like
EfficientNet-Lite or ShuffleNet ), we provide the student model with a strong inductive
bias for low-resource operation. The student model will start with this architecture, and it
may be further slimmed or adjusted during the compression process (e.g., reducing
width or depth) as needed. Using such an efficient backbone is crucial – for instance,
MobileNet-v3 is already targeted at mobile with 16% of the parameters of a typical CNN,
yet offers competitive accuracy ([2205.00211] DefakeHop++: An Enhanced Lightweight
Deepfake Detector). This gives us a head start in making the model lightweight. We will
initialize the student model with random weights (or possibly pre-trained ImageNet
weights if available for the chosen architecture) before the distillation process. The
capacity gap between teacher and student (teacher being larger) ensures the student
has room to learn compressed knowledge. Notably, the student’s architecture will be
kept flexible: if the distilled student underperforms, we may iterate on the architecture
(e.g., slightly increase its size or change layers) to hit the best trade-off between size
and accuracy.
```
```
Teacher Model Training: We will train the teacher model on the training dataset in a
supervised manner using the ground truth labels (real or fake). Standard training
protocols will be followed: using a binary cross-entropy (or focal loss) as the loss function
for classification, with an optimizer like Adam. The teacher model will learn to output a
high confidence for real vs fake classification. We will monitor its performance on the
validation set and ensure it achieves strong accuracy (aiming for near state-of-the-art
results on the dataset). Early stopping or checkpointing will be used to prevent
overfitting. The purpose of this stage is to obtain a high-accuracy reference model that
encapsulates the knowledge of detecting deepfakes.
Knowledge Distillation (Teacher-to-Student Transfer): In the second stage, we train
the student model under the guidance of the teacher model , employing knowledge
distillation. Knowledge distillation is a model compression technique where
“knowledge” from a larger pre-trained model (teacher) is transferred to a smaller
model (student) (A Comparative Analysis of Compression and Transfer Learning
Techniques in DeepFake Detection Models). This typically involves training the student
on the same dataset, but instead of relying only on the hard labels (real/fake ground
truth), the student is also penalized based on the soft predictions of the teacher.
Concretely, we will use a distillation loss that combines two terms: (1) a standard
classification loss (cross-entropy) between the student’s predictions and the true
labels, and (2) a distillation loss between the student’s output probabilities and the
teacher’s output probabilities for each input (A novel model compression method based
on joint distillation for deepfake video detection | OpenReview) (A novel model
compression method based on joint distillation for deepfake video detection |
```

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

The final step is to prepare the optimized model for deployment on mobile devices, which
involves **model quantization** and integration into a mobile app framework. Quantization will
further compress the model by reducing the numerical precision of the model’s weights (and
possibly activations), typically from 32-bit floating point to 8-bit integers. This can **shrink
memory and storage requirements by ~75%** and speed up inference using integer
arithmetic (What Is int8 Quantization and Why Is It Popular for Deep Neural Networks? -
MATLAB & Simulink) (What Is int8 Quantization and Why Is It Popular for Deep Neural
Networks? - MATLAB & Simulink). We will perform **8-bit post-training quantization** on the
pruned student model. This means converting the weights to int8 and calibrating the
activations’ dynamic ranges using a representative dataset (usually a small set of training or

```
balance across layers, we may remove an equal percentage of parameters per layer, as
a form of local pruning (A Comparative Analysis of Compression and Transfer Learning
Techniques in DeepFake Detection Models), ensuring no single layer becomes a
bottleneck. Alternatively, if supported by our framework, structured pruning (removing
entire filters or channels) could be used for a more hardware-friendly outcome (since
removed neurons yield speedups). We will decide the pruning granularity based on what
yields the best efficiency gains with minimal accuracy drop.
Pruning Rate and Iterative Pruning: We will determine a pruning rate (for example,
removing 20% of weights initially) and then iteratively prune and retrain. It is often
effective to prune gradually – remove a small percentage of weights and then fine-tune
the model to recover performance , rather than prune too aggressively in one go. Fine-
tuning after pruning allows the remaining weights to adjust and compensate for the lost
connections (A Comparative Analysis of Compression and Transfer Learning Techniques
in DeepFake Detection Models). We will prune the network in one or multiple stages,
each followed by a short re-training (with the same distillation or classification loss on the
training set) to regain any slight accuracy drop due to pruning. This prune-and-fine-tune
cycle will continue until we reach a desirable trade-off: a substantially smaller model that
still meets our accuracy target. For instance, we might target a 30-50% reduction in the
number of parameters or FLOPs from the distilled student. We will use the validation set
to ensure the accuracy after each pruning iteration is acceptable. If a severe accuracy
degradation is observed at any pruning level, we will stop or consider reverting the last
step.
Expected Outcome of Pruning: By the end of this step, the student model will be even
more compact. Pruning can dramatically shrink model size by removing redundancies
that the student didn’t actually need for the task. This contributes directly to faster
inference on mobile and lower memory usage. Research shows that a carefully pruned
model, followed by fine-tuning, can often retain accuracy very close to the original
model’s (A Comparative Analysis of Compression and Transfer Learning Techniques in
DeepFake Detection Models). We aim to validate that in our experiments – comparing
the pruned model’s accuracy to the unpruned student. We will document the final pruned
model’s size (parameter count) and performance.
```

validation samples) so that the quantized model maintains accuracy (How the Deep
Learning benchmark performed for 16 bit and for 8 bit ...) (A Comparative Analysis of
Compression and Transfer Learning Techniques in DeepFake Detection Models). Modern
tools like **TensorFlow Lite** or **PyTorch Mobile** provide built-in support for such quantization
and can often do this conversion with minimal loss in model accuracy. By quantizing, we
expect only a minor drop (often just 1-2% in accuracy, if any) but a significant gain in
inference speed and a reduction in model size (the model file becomes quarter the size of
the 32-bit version) (What Is int8 Quantization and Why Is It Popular for Deep Neural
Networks? - MATLAB & Simulink) (What Is int8 Quantization and Why Is It Popular for Deep
Neural Networks? - MATLAB & Simulink). If we find the accuracy drop is larger than
expected, we might employ quantization-aware training (fine-tuning the model with fake
quantization in the loop) to better preserve performance – though this adds complexity and
may not be necessary given the relatively simple classification task.

**Mobile Integration:** Once quantized, we will integrate the model into a mobile application
prototype. For Android, we can use TensorFlow Lite to load the .tflite quantized model.
For iOS, Core ML conversion is an option. The app will take video input from the device’s
camera or video files and run the deepfake detection model on each frame (or every few
frames) in real time. Because our model is extremely lightweight at this stage (small number
of parameters, int8 precision), it should be capable of near real-time inference even on a
mobile CPU. We will test the inference speed (frames per second) on a typical mid-range
smartphone to verify this. Additionally, we will measure the model’s memory footprint. The
goal is to ensure the model can process video frames on-device without noticeable lag or
excessive battery drain. If needed, we may optimize further by leveraging device
accelerators (DSP/NPU) via frameworks or by adjusting the model (for example, ensuring
ops are supported by the mobile GPU delegate).

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
familiarize with the NCNN framework documentation. Simultaneously, refine the thesis
proposal (this document) based on initial findings and feedback, clearly defining the
project scope and plan.
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

**Month 5: Model Optimization and Conversion** – Once a satisfactory model is trained,
optimize it for mobile deployment. This includes converting the model to a portable
format and integrating with NCNN. Use tools to convert the model (e.g., export to ONNX,
then use NCNN’s conversion utility to generate .param and .bin files). Perform
quantization if necessary and measure any drop in accuracy versus size/speed gains.
Test the NCNN model on a desktop or emulator to ensure it produces the same outputs
as the original (verifying correctness after conversion). Iterate on this process if any
issues arise (for example, if a layer is unsupported, replace it and retrain or modify the
model). By the end of this month, have the model running in NCNN outside the Android
app (e.g., in a simple C++ test environment), and ready to be embedded into the app.
**Month 6: Android Application Development** – Develop the Android app that will host
the deepfake detection functionality. Set up the Android Studio project, include the
NCNN library, and write the JNI bridge code to load the model and run inference. Design
and implement the user interface, ensuring it is intuitive. Integrate image selection or
camera capture features for users to input media. Connect the UI with the detection
logic: when the user submits an image or video, run the NCNN model on it and return
the result to the UI. Conduct preliminary testing with sample images to verify end-to-end
functionality (correct loading of model, inference, and result display).

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

