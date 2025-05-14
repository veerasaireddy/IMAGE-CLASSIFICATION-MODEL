# IMAGE-CLASSIFICATION-MODEL

COMPANY : CODTECH IT SOLUTIONS

NAME : BUSIREDDY VEERA SAI REDDY

INTERN ID : CT04DM982

DOMAIN : MACHINE LEARNING

DURATION : 4 WEEKS

MENTOR : NEELA SANTHOSH KUMAR



📄 Task 3: Image Classification Using Convolutional Neural Networks (CNN)
🔍 Introduction
As part of the CodTech Machine Learning Internship, Task 3 focuses on one of the most widely used applications of deep learning — image classification. The aim is to develop a Convolutional Neural Network (CNN) using either TensorFlow or PyTorch to accurately classify images based on their content. CNNs are at the heart of modern computer vision systems, powering applications from face recognition to autonomous vehicles. In this task, I used TensorFlow due to its ease of use, strong documentation, and seamless integration with Keras APIs.

🧠 Objective
The objective of this task is to build a functional deep learning model that can recognize patterns in image data and correctly classify unseen images. This involves designing a CNN architecture, training it on labeled data, validating its performance, and finally evaluating it on a test dataset. The deliverable is a trained model along with its performance metrics — primarily accuracy on the test set.

📦 Dataset Used: MNIST
To implement and validate the CNN, the MNIST dataset was used. This dataset contains 70,000 grayscale images of handwritten digits (0–9), where each image is 28x28 pixels. It is a well-known benchmark in the machine learning community and is ideal for beginners due to its simplicity and reliability. The dataset is already split into a training set of 60,000 images and a test set of 10,000 images, making it suitable for both training and evaluation.

🧹 Data Preprocessing
The first step involved normalizing the pixel values from the range [0, 255] to [0, 1] by dividing each value by 255. This scaling helps in faster convergence during training. Additionally, the image data was reshaped to include a channel dimension, converting it from a 2D format (28x28) to a 4D tensor (batch_size, height, width, channels). The labels, which are originally integer class values (e.g., 0, 1, 2, ..., 9), were one-hot encoded using TensorFlow utilities to match the softmax output from the model.

🏗️ Model Architecture
The CNN architecture was built using the Sequential API of TensorFlow's Keras module. The network structure included:

Convolutional Layer 1: 32 filters of size 3x3 with ReLU activation to detect low-level features.

MaxPooling Layer 1: 2x2 pool size to reduce spatial dimensions.

Convolutional Layer 2: 64 filters to detect more complex features.

MaxPooling Layer 2: Further reduces the feature maps.

Flatten Layer: Converts the 2D feature maps into a 1D vector.

Dense Layer: Fully connected layer with 128 neurons and ReLU activation.

Output Layer: 10 neurons (for digits 0–9) with softmax activation to output probabilities.

⚙️ Training and Optimization
The model was compiled using:

Loss Function: categorical_crossentropy — suitable for multi-class classification.

Optimizer: Adam — an efficient gradient descent optimization algorithm.

Metrics: accuracy — to monitor the performance during training.

The model was trained for 5 epochs with the validation data provided by the test set. The training process showed consistent improvements in both training and validation accuracy.

📊 Evaluation
After training, the model was evaluated on the test set using the evaluate() function. The CNN achieved an impressive accuracy of over 98%, demonstrating its ability to generalize well to unseen data. This high performance is typical for CNNs on MNIST, given its clean and balanced dataset.

✅ Final Output
The final deliverable is a Jupyter Notebook titled task3_cnn_image_classification.ipynb, containing:

Data loading and preprocessing

CNN model construction

Model training and validation

Test accuracy evaluation

The notebook is modular, well-commented, and ready for submission as part of the internship requirements.

🧾 Conclusion
This task offered valuable experience in building deep learning models for image classification. It provided hands-on exposure to TensorFlow, data preprocessing techniques, CNN architecture design, and model evaluation. By working through this task, I developed a deeper understanding of how CNNs learn spatial hierarchies in image data and how they are trained for real-world image recognition problems. This foundational project sets the stage for more complex computer vision applications in the future.



### OUTPUT  :

<img width="1039" alt="Image" src="https://github.com/user-attachments/assets/2c8ea687-8ff7-46e2-908a-f57841e36341" />
