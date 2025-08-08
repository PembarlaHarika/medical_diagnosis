#import required libraries
import os
import tensorflow

#import ImageDatagenerator class for image preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#step 1: Define base dataset directory
#__file__ is the current script file
#os.path.abspath(__file__) gets the absolute path of the "dataset" folder
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))

#step 2: define sun-directories for training and validation , and test data
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

#step 3: Create ImageDataGenerator for training data
#it rescaies pixel valuesand applies data augmentation (rotation, flip)
train_datagen = ImageDataGenerator(
    rescale=1./255,       #Normalize pixel values to [0-255 -> 0-1]
    rotation_range=15,    # randomly rotate images by up to 15 degrees
    horizontal_flip=True, #randomly flip images horizontall

)
#step 4: ImageDataGenerator for validation data
val_datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values

#step 5: ImageDataGenerator for test data
test_datagen = ImageDataGenerator(rescale=1./255)  

#step 6: Create data loaders from the folders using flow_from_directory()
#This

#Load training data
train_generator = train_datagen.flow_from_directory(
    train_dir,               #folder path
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,            # Load 32 images per batch
    class_mode='categorical',  # Use one-hot encoded labels (for multi-class)
    shuffle=True              # Shuffle  data for each epoch
)

#Load validation data
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),  
    batch_size=32,            
    class_mode='categorical',  
    shuffle=False             # No need to shuffle validation data
)

#Load test data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),  
    batch_size=32,            
    class_mode='categorical',  
    shuffle=False             # keep order evalution/prediction
)

#step 7: Fetch a batch of images and labels to check
images, labels = next(train_generator)

#print the shape of the loaded batch
print("Batch shape:", images.shape)
print("Labels shape:", labels.shape)

##(10,2)

