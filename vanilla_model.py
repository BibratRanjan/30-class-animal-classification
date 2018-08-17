from keras import layers
from keras import models
import os
from keras import optimizers
from model_prediction import logic
import glob
import cv2, numpy as np
import csv
import re

test_flag = True 

nb_classes = 30
base_dir = './Dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


def architecture(weights = None):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(nb_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3), metrics=['acc'])
    return model

def write_to_csv(image_ids, output):
    with open("./DL_Beginner/meta-data/sample_submission.csv", "r") as f:
        reader = csv.reader(f)
        csv_cols = next(reader)
    
    csv_file = "output.csv"
    output = output.tolist()
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_cols)    
            for i in range(len(image_ids)):
                writer.writerow([image_ids[i]] + output[i])
    except IOError:
        print("I/O error") 

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
checkpointer = ModelCheckpoint(filepath = './checkpoints/vanilla_model.h5', verbose=1, save_best_only=True)
tb = TensorBoard(log_dir='./logs')
early_stopper = EarlyStopping(patience=3)


if test_flag  & os.path.exists('./checkpoints/vanilla_model.h5') :
    model = architecture(models.load_model('./checkpoints/vanilla_model.h5'))
    X_test = []
    image_ids = []
    temp_x_test_paths = []    
    
    """Sorting the image paths"""
    X_test_paths = glob.glob(os.path.join(test_dir, '*.jpg'))
    for path in X_test_paths:
        temp = re.split('/|-', path)
        temp[-1] = (temp[-1].split('.')[0].zfill(4)) + '.' + temp[-1].split('.')[1]
        path = os.path.join(temp[0], temp[1],  temp[2], temp[3] + '-' + temp[4])
        temp_x_test_paths.append(path)    
    temp_x_test_paths = sorted(temp_x_test_paths)
    X_test_paths = []
    for path in temp_x_test_paths:
        temp = re.split('/|-', path)
        temp[-1] = temp[-1].strip('0')
        path = os.path.join(temp[0], temp[1],  temp[2], temp[3] + '-' + temp[4])
        X_test_paths.append(path)
    
    for path in X_test_paths:
        image_id = path.split(os.path.sep)[3]
        image = cv2.resize(cv2.imread(path), (150, 150), interpolation = cv2.INTER_CUBIC)
        
        X_test.append(image)
        image_ids.append(image_id)
        
    X_test = np.array(X_test)
    output = model.predict_proba(X_test, verbose = 1)
    
    write_to_csv(image_ids, output)
    
else:
    model = architecture()    

    from keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    batch_size = 20
    steps_per_epoch = 28800 // batch_size #960 samples per class
    validation_steps = 7200 // batch_size #240 samples per class
    
    
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=batch_size, class_mode='categorical')
    validation_generator = val_datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=batch_size, class_mode='categorical')
    
     
    
    history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=7, verbose=1, 
                                  callbacks=[tb, early_stopper, checkpointer], validation_data=validation_generator, 
                                  validation_steps=validation_steps, workers=4)
    
    #model.save('./checkpoints/vanilla_model.h5')
    
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()