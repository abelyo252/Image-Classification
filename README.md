# Image-Classification

This Colab notebook showcases an image classification model trained to classify images into different categories. The model has been trained on a dataset comprising images from various classes, including anacephaly, congi, down_syndrome, encephalocele, and placenta.

![Image](https://upload.wikimedia.org/wikipedia/commons/c/c7/CRL_Crown_rump_length_12_weeks_ecografia_Dr._Wolfgang_Moroder.jpg?Ultrasound1163529084)

 This project aims to showcase the process of building an image classification model and provide a starting point for further exploration and experimentation.

 # Dataset
 The dataset used for training, validation, and testing the model is loaded using the tf.keras.utils.image_dataset_from_directory function. It is organized in a directory structure where each class has its own subdirectory containing the corresponding images. The dataset is divided into training, validation, and test sets using a **70-20-10** split.

 ![Image](https://github.com/abelyo252/Image-Classification/blob/main/performance/dataset2.png)

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/abelyo252/Image-Classification.git
   cd image-classification
   ```
   



2. Install the required dependencies:
   ```shell
   pip install -r requirements.txt
   ```
    

3. Usage

    Prepare your dataset:
        Organize your images into separate directories for each class.
        use this notebook to train based on your custom data, but make 
        sure change hyper-parameter of this model.

4. Inference:
   
   ```python
       import tensorflow as tf
       import numpy as np
       import cv2
   
       # Load the trained model
       model = tf.keras.models.load_model('load_your_model.h5')
   
       # Load and preprocess the input image
       image = tf.keras.preprocessing.image.load_img('path/to/image.jpg', target_size=(224, 224))
       image = tf.keras.preprocessing.image.img_to_array(image)
   
       # Perform inference
       image = cv2.resize(image, (256, 256))
       img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       # Preprocess the image
       input_image = np.expand_dims(img_rgb, axis=0)  # Add a batch dimension
       input_image = input_image / 255.0  # Normalize the image
       
       # Make predictions
       predictions = model.predict(input_image)
       # Get the predicted class labels
       predicted_labels = tf.argmax(predictions, axis=1)
       
       # Map the predicted labels to class names
       predicted_class_names = [class_names[label] for label in predicted_labels]
       # Print the predicted class names
       print("Predicted Class Names:", predicted_class_names)
   
   
       Replace 'path/to/image.jpg' with the path to your input image.
   ```


  
---
## Support

You can ask questions and join the development discussion:

- @ Telegram t.me/@benyohanan

---

## Opening an issue

You can also post **bug reports and feature requests** (only)
in [GitHub issues](https://github.com/ab).


---

## Opening a PR

We welcome contributions! Before opening a PR, please read
[our contributor guide](https://github.com//CONTRIBUTING.md).
