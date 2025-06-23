import os
import numpy as np
import random
from PIL import Image,ImageEnhance

#keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from sklearn.utils import shuffle


#load dataset

train_dir = 'brain_tumor_dataset\Training'
test_dir = 'brain_tumor_dataset\Testing'

train_paths = []
train_labels = []

for label in os.listdir(train_dir):
  for image in os.listdir(os.path.join(train_dir,label)):
    train_paths.append(os.path.join(train_dir,label,image))
    train_labels.append(label)

train_paths,train__labels  = shuffle(train_paths,train_labels)

test_paths = []
test_labels = []

for label in os.listdir(test_dir):
  for image in os.listdir(os.path.join(test_dir,label)):
    test_paths.append(os.path.join(test_dir,label,image))
    test_labels.append(label)


test_paths,test__labels  = shuffle(test_paths,test_labels)


#DATA VISUALISATION

import random
import matplotlib.pyplot as plt
plt.show() #WHENEVER TO SHOW 


random_indices = random.sample(range(len(train_paths)),10)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for i, index in enumerate(random_indices):
    img_path=train_paths[index]
    img = Image.open(img_path)
    img = img.resize((128, 128))

    axes[i].imshow(img)
    axes[i].axis("off")
    axes[i].set_title(f"Label :{train_labels[index]}",fontsize=20)



plt.tight_layout()
plt.show()


#IMAGE PREPROCESSING

def augment_image(image):
  image = Image.fromarray(np.uint8(image))
  image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8,1.2))
  image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8,1.2))
  image = np.array(image)/255.0
  return image

def open_images(paths):
  images = []
  for path in paths:
    img = load_img(path,target_size=(IMAGE_SIZE,IMAGE_SIZE))
    img = augment_image(img)
    images.append(img)
  return np.array(images)


def encode_label(labels):
  unique_labels = os.listdir(train_dir)
  encoded = [unique_labels.index(label) for label in unique_labels]
  return encoded


def datagen(paths, labels, batch_size=12,epochs=1):
  for _ in range(epochs):
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            batch_images = open_images[batch_paths]
            batch_labels = labels[i:i + batch_size]
            batch_labels = encode_label(batch_size)
            yield batch_images, batch_labels


#EPOCH RUN

# ... (previous imports and data loading)

# Define IMAGE_SIZE before using it
IMAGE_SIZE = 128

# ... (augment_image, open_images, encode_label functions)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Import ImageDataGenerator
import os # Import the os module
import math # Import the math module for ceil

# Assuming train_paths and train_labels are already loaded from previous cells

base_model = VGG16(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),include_top=False, weights='imagenet')

for layer in base_model.layers:
  layer.trainable = False

unfreeze_layers_count = 13
for layer in base_model.layers[-unfreeze_layers_count:]:
    layer.trainable = True


model = Sequential()
model.add(Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
num_classes = len(os.listdir(train_dir))
model.add(Dense(num_classes,activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0001), # Further reduced learning rate to 0.000001
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

batch_size = 20
epochs =10 

# Use ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Rescale images to [0, 1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create a generator
train_generator = train_datagen.flow_from_directory(
    'brain_tumor_dataset/Training',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=True
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'brain_tumor_dataset/Testing',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False
)

#www.linkedin.com/in/yogitabisht2003
# Calculate steps_per_epoch using math.ceil to include the last partial batch
steps_per_epoch = math.ceil(train_generator.samples / batch_size)


# The history variable is assigned here by running model.fit
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Get class names from folders
classes = sorted(os.listdir(train_dir))

# Compute weights using actual training labels (not directory names)
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(train_labels), 
    y=train_labels
)

# Map class indices to weights
class_weight_dict = {i: class_weights[i] for i in range(len(classes))}

# Train the model with class weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Get class names from folders
classes = sorted(os.listdir(train_dir))

# Compute weights using actual training labels (not directory names)
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(train_labels), 
    y=train_labels
)

# Map class indices to weights
class_weight_dict = {i: class_weights[i] for i in range(len(classes))}

# Train the model with class weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Get class names from folders
classes = sorted(os.listdir(train_dir))

# Compute weights using actual training labels (not directory names)
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(train_labels), 
    y=train_labels
)

# Map class indices to weights
class_weight_dict = {i: class_weights[i] for i in range(len(classes))}

# Train the model with class weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Get class names from folders
classes = sorted(os.listdir(train_dir))

# Compute weights using actual training labels (not directory names)
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(train_labels), 
    y=train_labels
)

# Map class indices to weights
class_weight_dict = {i: class_weights[i] for i in range(len(classes))}

# Train the model with class weights

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Get class names from folders
classes = sorted(os.listdir(train_dir))

# Compute weights using actual training labels (not directory names)
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(train_labels), 
    y=train_labels
)

# Map class indices to weights
class_weight_dict = {i: class_weights[i] for i in range(len(classes))}

# Train the model with class weights

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Get class names from folders
classes = sorted(os.listdir(train_dir))

# Compute weights using actual training labels (not directory names)
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(train_labels), 
    y=train_labels
)

# Map class indices to weights
class_weight_dict = {i: class_weights[i] for i in range(len(classes))}

# Train the model with class weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Get class names from folders
classes = sorted(os.listdir(train_dir))

# Compute weights using actual training labels (not directory names)
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(train_labels), 
    y=train_labels
)

# Map class indices to weights
class_weight_dict = {i: class_weights[i] for i in range(len(classes))}

# Train the model with class weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Get class names from folders
classes = sorted(os.listdir(train_dir))

# Compute weights using actual training labels (not directory names)
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(train_labels), 
    y=train_labels
)

# Map class indices to weights
class_weight_dict = {i: class_weights[i] for i in range(len(classes))}



from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Get class names from folders
classes = sorted(os.listdir(train_dir))

# Compute weights using actual training labels (not directory names)
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(train_labels), 
    y=train_labels
)

# Map class indices to weights
class_weight_dict = {i: class_weights[i] for i in range(len(classes))}

# Train the model with class weights
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,
    class_weight=class_weight_dict
)

#model training history

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.grid(True)

# Check if the keys exist before plotting
if 'sparse_categorical_accuracy' in history.history:
    plt.plot(history.history['sparse_categorical_accuracy'], '.g-', linewidth=2)
else:
    plt.plot(history.history['accuracy'], '.g-', linewidth=2)  # Fallback if key is 'accuracy'

plt.plot(history.history['loss'], '.r-', linewidth=2)

plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.xticks(range(len(history.history['loss'])))
plt.legend(['Accuracy', 'Loss'], loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()


#PLOTTING 

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


def encode_label(labels):
    unique_labels = sorted(os.listdir(train_dir))  # Only 4 classes
    return [unique_labels.index(label) for label in labels]

# Preprocess test images and labels
test_labels_encoded = test_generator.classes  # This gives all 1311 labels already encoded
test_images = test_generator  # Already batched and normalized


# Make predictions
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

print("Classification Report:")
print(classification_report(test_labels_encoded, predicted_labels))

# Check sizes for verification
print(len(test_labels_encoded), predictions.shape)
print(predicted_labels.shape)


# Precision, Recall, F1 Score
precision = precision_score(test_labels_encoded, predicted_labels, average='weighted')
recall = recall_score(test_labels_encoded, predicted_labels, average='weighted')
f1 = f1_score(test_labels_encoded, predicted_labels, average='weighted')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Confusion matrix
conf_matrix = confusion_matrix(test_labels_encoded, predicted_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=os.listdir(train_dir),
            yticklabels=os.listdir(train_dir))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


#MODEL CONFUSION PLOT

conf_matrix = confusion_matrix(test_labels_encoded,np.argmax(predictions,axis=1))
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues',xticklabels=os.listdir(train_dir),yticklabels=os.listdir(train_dir))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show

#ROC CURVE PLOT

#binarize3 testlabels and predictions for multi-class ROC
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.preprocessing import label_binarize

test__labels_bin = label_binarize(test_labels_encoded,classes=np.arange(len(os.listdir(train_dir))))
test_predictions_bin = predictions

#compute ROC curve amd ROC AUC for each class
from sklearn.metrics import roc_curve, auc

fpr,tpr,roc_auc = {},{},{}
for i in range(len(os.listdir(train_dir))):
   fpr[i],tpr[i],_ = roc_curve(test__labels_bin[:,i],test_predictions_bin[:,i])
   roc_auc[i]=auc(fpr[i],tpr[i])

#Plot ROC curve
plt.figure(figsize=(10,8))
for i in range(len(os.listdir(train_dir))):
   plt.plot(fpr[i],tpr[i],label=f'class{i} (AUC={roc_auc[i]:2F})')

plt.plot([0,1],[0,1],linestyle='--',color='gray')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

import os
print(os.listdir())

# SAVE THE ENTIRE MODEL
model.save('my_model.h5')
import tensorflow as tf

from tensorflow.keras.models import load_model
#load the trained model
model = tf.keras.models.load_model('my_model.h5')

from tensorflow.keras.models import load_model

model = load_model("my_model.h5")
model.summary()  # prints model architecture

for layer in model.layers:
    try:
        print(layer.name, layer.get_output_shape_at(0))
    except:
        print(layer.name, "Shape not available")



#
from keras.preprocessing.image import load_img,img_to_array

class_labels = ['pituitary','glioma','notumor','meningioma']

def detect_and_display(image_path,model):
   try:
      #load image
      img = load_img(img_path,target_size=(128,128))
      img_array = img_to_array(img)
      img_array = np.expand_dims(img_array,axis=0)

      #prediction
      predictions = model.predict(img_array)
      predicted_class_index = np.argmax(predictions)
      confidence_score = np.max(predictions,axis=1)[0]

      #determine the class
      if class_labels[predicted_class_index]=='notumor':
         return "NO TUMOR DETECTED"
      else:
         result=f"Tumor: {class_labels[predicted_class_index]}"

      #display
      plt.imshow(load_img(img_path))
      plt.axis('off')
      plt.title(f"{result}(Confidence:){confidence_score * 100:.2f}")
      plt.show()
      
   except Exception as e:
      print("Error processing the image:",str(e))

#EXAMPLE

#for no tumor
image_path ='brain_tumor_dataset\Testing\notumor'
detect_and_display(image_path,model)

#for Glioma 
image_path ='brain_tumor_dataset\Testing\glioma'
detect_and_display(image_path,model)

#for Meningioma
image_path ='brain_tumor_dataset\Testing\meningioma'
detect_and_display(image_path,model)

#for Pituitary
image_path ='brain_tumor_dataset\Testing\pituitary'
detect_and_display(image_path,model)
   

