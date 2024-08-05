
# Traffi Light Classifier Script (VGG16 Transfer Learning)
## Description
This project provides a traffic light classifier algorithm using the VGG16 pre-trained model for model training, validation and testing.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation
Open [Google Colaboratory](https://colab.research.google.com/) and sign up to create an account.

## Usage
To use the script, you need to update the file paths with appropriate paths.

    1. E.g. Update the train_path, and test_path variables in the script with the appropriate paths.
    2. Run the script:

### Code Snippets
Importing libraries and pre-trained model:

    from keras.layers import Input, Lambda, Dense, Flatten
    from keras.models import Model
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg16 import preprocess_input
    from keras.preprocessing import image
    from keras.models import Sequential
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import numpy as np
    from glob import glob
    import matplotlib.pyplot as plt

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

Define image size:

    IMAGE_SIZE = [224, 224]

Mount Google Drive:

    from google.colab import drive
    drive.mount('/content/drive')

List the contents of the 'drive' directory:

    !ls '/content/drive'

Give dataset paths:

    train_path = '/path/to/your/train/folder'
    test_path = '/path/to/your/test/folder'

Visual inspection of the dataset from a specified folder :

    from PIL import Image
    import os
    from IPython.display import display
    from IPython.display import Image as _Imgdis
    # creating a object


    folder = train_path+'/Green_Circle'


    onlyGreenCirclefiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    print("Working with {0} images".format(len(onlyGreenCirclefiles)))
    print("Image examples: ")


    for i in range(10):
        print(onlyGreenCirclefiles[i])
        display(_Imgdis(filename=folder + "/" + onlyGreenCirclefiles[i], width=240, height=240))

Loading of VGG16 pre-trained model on the ImageNet dataset:

    vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

Input layer for building a new model:

    vgg.input

Freeze the layers of the VGG16 model to prevent the pre-trained weights from being updated during training:

    for layer in vgg.layers:
    layer.trainable = False

Count and print the number of folders in a specified directory:

    folders = glob('/path/to/your/folder/*')
    print(len(folders))

 Print model summary:
    
    x = Flatten()(vgg.output)
    prediction = Dense(len(folders), activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=prediction)
    model.summary()

Initialise the Adam optimizer and compile the model with categorical cross-entropy loss (suitable for multi-class classification):

    from keras import optimizers


    adam = optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                optimizer=adam,
                metrics=['accuracy'])

Apply the configured ImageDataGenerator to create data generators for training:

    train_datagen = ImageDataGenerator()

Apply the configured ImageDataGenerator to create data generators for testing and use preprocess_input from pre-trained model to ensure that the input images are preprocessed in the same way as the images were when the model was originally trained:

    test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
    )

Load images from a directory, apply transformations, and prepare them for training:
    
    train_set = train_datagen.flow_from_directory(train_path,
                                                  target_size = (224, 224),
                                                  batch_size = 32,
                                                  class_mode = 'categorical')

Load images from a directory, apply transformations, and prepare them for testing:

    test_set = test_datagen.flow_from_directory(test_path,
                                                target_size = (224, 224),
                                                batch_size = 32,
                                                class_mode = 'categorical',
                                                shuffle = False)

Model training:

    from datetime import datetime
    from keras.callbacks import ModelCheckpoint



    checkpoint = ModelCheckpoint(filepath='/file/path/to/save/your/trained/model',
                                verbose=2, save_best_only=False)

    callbacks = [checkpoint]

    start = datetime.now()

    model_history=model.fit(
    train_set,
    validation_data=test_set,
    epochs=10,
    steps_per_epoch=5,
    validation_steps=32,
        callbacks=callbacks ,verbose=2)


    duration = datetime.now() - start
    print("Training completed in time: ", duration)

Load saved model:

    from keras.models import load_model
    model = load_model('/path/to/your/saved/model.h5')

Evaluating the model and making predictions on the test set:

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_set)
    print("Test accuracy:", test_accuracy)

    # Predict classes for the test set
    test_pred = model.predict(test_set)

    # Get the predicted classes
    test_pred_classes = np.argmax(test_pred, axis=1)

    # Get the true classes
    true_classes = test_set.classes

Display the classification report, accuracy, predicted and actual classes, and predicted probabilities:

    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score

    # Display the classification report
    print(classification_report(true_classes, test_pred_classes))

    # Display the accuracy
    print("Accuracy:", accuracy_score(true_classes, test_pred_classes))

    # Display the predicted and actual classes
    for i in range(10):
    print("Predicted class:", test_pred_classes[i])
    print("Actual class:", true_classes[i])
    print("")

    # Display the predicted probabilities
    for i in range(10):
    print("Predicted probabilities:", test_pred[i])
    print("")

Print index of true classes:

    print(true_classes)

Print index of predicted classes:

    print(test_pred_classes)

Display test results in a confusion matrix:

    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    # Display the confusion matrix
    conf_mat = confusion_matrix(true_classes, test_pred_classes)
    df_cm = pd.DataFrame(conf_mat, range(len(folders)), range(len(folders)))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    plt.show()

Save test results in a CSV file:

    # Save the results in a CSV file
    import csv

    with open('contest6.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image", "Actual Class", "Predicted Class"])
        for i in range(len(test_set.filenames)):
            writer.writerow([test_set.filenames[i], true_classes[i], test_pred_classes[i]])

Plot training, validation, and test accuracy values in a line graph:

    # Plot training, validation, and test accuracy values
    plt.plot(model_history.history['accuracy'], label='Train')
    #plt.plot([test_accuracy]*len(model_history.history['accuracy']), label='Test')
    plt.plot(model_history.history['val_accuracy'], label='Validation')
    plt.title('CNN Model accuracy values')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

## License
This project is licensed under the [MIT License](https://www.mit.edu/~amini/LICENSE.md).



