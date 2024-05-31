## Deliverables for Deep Learning for Computer Vision Course

This repository contains the following deliverables for the Deep Learning for Computer Vision course, part of the Master's degree in Data Science and Machine Learning:

### Assignment 1: Image Classification Systems

**Objectives:**
- Understand an Image Classification system and the data-driven approach (training and prediction stages).
- Understand the training/validation/test split and the use of validation data for hyperparameter tuning.
- Develop the skill to write efficient vectorized code with numpy.
- Implement and apply a k-Nearest Neighbors (kNN) classifier.
- Implement and apply a Softmax classifier.
- Implement and apply a two-layer Neural Network classifier.
- Understand the differences and respective advantages and disadvantages of these classifiers.
- Gain a basic understanding of performance improvements using high-level representations as opposed to raw pixel values (e.g., color histograms, Histogram of Gradients (HOG) features, etc.).

**Content:**
- Q1: k-Nearest Neighbors Classifier
- Q2: Implement a Softmax Classifier
- Q3: Two-Layer Neural Network
- Q4: Higher-level Representations: Image Features

**Files:**
- `assignment1/knn.ipynb`
- `assignment1/softmax.ipynb`
- `assignment1/two_layer_net.ipynb`
- `assignment1/features.ipynb`

### Assignment 2: Backpropagation and Training Neural Networks

**Objectives:**
- Understand Neural Networks and layer-wise architecture.
- Understand and implement the Backpropagation algorithm (vectorized).
- Implement various update techniques to optimize Neural Networks.
- Understand and implement Batch Normalization for training deep networks.
- Implement Dropout as a regularization method.
- Understand the architecture of Convolutional Neural Networks (CNNs) and practice training them.
- Gain experience with one of the main deep learning libraries, PyTorch.

**Content:**
- Q1: Fully Connected Neural Networks
- Q2: Batch Normalization
- Q3: Dropout
- Q4: Convolutional Networks
- Q5: PyTorch on CIFAR-10

**Files:**
- `assignment2/FullyConnectedNets.ipynb`
- `assignment2/BatchNormalization.ipynb`
- `assignment2/Droput.ipynb`
- `assignment2/ConvolutionalNetworks.ipynb`
- `assignment2/PyTorch.ipynb`

### Assignment 3: Advanced Neural Network Architectures

**Objectives:**
- Implement language models and apply them to image captions on the COCO dataset.
- Train an adversarial generative network to generate images similar to those in the training dataset.
- Work on self-supervised learning to automatically learn visual representations from an unlabeled dataset.
- Understand and implement RNN and Transformer networks, and combine them with CNNs for image captioning.
- Explore various applications of image gradients, including saliency maps, adversarial images, and class visualization.
- Train and implement Generative Adversarial Networks (GANs) to produce images similar to samples from a dataset.
- Utilize self-supervised learning techniques to assist with image classification tasks.

**Content:**
- Q1: Image Captioning with Vanilla RNNs
- Q2: Image Captioning with Transformers
- Q3: Network Visualization: Saliency Maps, Class Visualization, and Adversarial Images
- Q4: Generative Adversarial Networks (GANs)
- Q5: Self-Supervised Learning for Image Classification

**Files:**
- `assignment3/RNN_Captioning.ipynb`
- `assignment3/Transformer_Captioning.ipynb`
- `assignment3/Network_Visualization.ipynb`
- `assignment3/Generative_Adversarial_Networks.ipynb`
- `assignment3/Self_Supervised_Learning.ipynb`

### Final Assignment: ControlNet Paper Summary and Presentation

**Summary and Presentation of the ControlNet Paper:**
- **Paper:** Zhang, L., Rao, A., & Agrawala, M. (2023). Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 3836-3847).
- **Links:**
  - [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf)
  - [GitHub Repository](https://github.com/lllyasviel/ControlNet)

**Files:**
- `final_assignment/ControlNet.pdf`
