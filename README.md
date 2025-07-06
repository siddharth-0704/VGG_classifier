# Brain Tumor Detection using VGG Classifier

This project implements a Convolutional Neural Network (CNN) based on the VGG architecture to classify brain MRI images as either containing a tumor or not. The model is trained on the Brain MRI Images for Brain Tumor Detection dataset, which is publicly available on Kaggle. The goal is to develop a reliable binary classification system that can assist in detecting brain tumors from MRI scans using deep learning techniques.

The dataset consists of MRI images categorized into two classes: "Yes" for images showing the presence of a brain tumor, and "No" for those without any tumor. To access and download the dataset, the Kaggle API is used within the notebook. The commands set up the Kaggle environment, authenticate the API key, and download the dataset directly. After downloading, the data is unzipped and organized into folders for training and validation.

The model follows the VGG design principles, utilizing multiple convolutional and max pooling layers to extract spatial features from the images. These are followed by fully connected layers and a final sigmoid activation function for binary classification. The architecture is implemented using TensorFlow and Keras, and may include regularization techniques such as dropout and batch normalization to improve generalization.

The training pipeline includes preprocessing the images by resizing them to a uniform shape, normalizing pixel values, and encoding the labels into binary format. The data is split into training and validation sets. The model is compiled with binary crossentropy loss and optimized using the Adam optimizer. Throughout the training process, metrics such as accuracy and loss are tracked and visualized.

After training, the model is evaluated on the validation set. Performance is measured using metrics like accuracy, precision, recall, and the confusion matrix. Visualizations such as accuracy/loss curves help analyze how well the model is learning. The notebook also includes functionality to test the model on single images using a custom preprocessing pipeline, allowing real-time prediction.

This project depends on libraries such as TensorFlow, NumPy, Matplotlib, scikit-learn, and pandas. These should be installed prior to running the notebook. The project directory contains the main training notebook, the downloaded dataset (organized into 'yes' and 'no' folders), the Kaggle API key file, and the saved model file.

The final model achieves strong performance in distinguishing between tumor and non-tumor MRI images, with validation accuracy approaching 98 percent. The inference time per image is low, making the model suitable for real-time use cases in research or educational settings.

Possible improvements include applying data augmentation to improve robustness, experimenting with transfer learning using pretrained VGG16 or VGG19 models, and deploying the trained model using a web interface such as Streamlit or Flask. Further evaluation on more diverse datasets would also help improve generalization and real-world applicability.

This project is intended for educational and research purposes. It demonstrates how CNNs can be applied to medical image classification, but it is not designed or validated for clinical use. For any form of deployment in healthcare settings, further testing and regulatory approvals would be necessary.
