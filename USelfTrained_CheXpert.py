# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Importing Libraries

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import confusion_matrix, roc_curve
from tensorflow.keras.metrics import AUC
# # %load_ext tensorboard
from tensorflow.keras.applications.densenet import DenseNet121
import horovod.tensorflow.keras as hvd
import time

# +
# # !rm -rf ./logs/fit/uself

# +
# Horovod training

hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
# -

# # Loading Dataset

# +
source_path = '.'
train_directory = os.path.join(source_path, 'CheXpert-v1.0/train')
validation_directory = os.path.join(source_path, 'CheXpert-v1.0/valid')

print(f"There are {len(os.listdir(train_directory))}")
print(f"There are {len(os.listdir(validation_directory))}")
# -

# Load train and valid labels
train_df = pd.read_csv(os.path.join(source_path, 'CheXpert-v1.0/train.csv'))
valid_df = pd.read_csv(os.path.join(source_path, 'CheXpert-v1.0/valid.csv'))

# # Creating DataFrames

train_df = train_df[['Path','Atelectasis','Cardiomegaly','Consolidation','Edema','Pleural Effusion',
                     'No Finding','Enlarged Cardiomediastinum', 'Lung Opacity','Lung Lesion','Pneumonia',
                     'Pneumothorax', 'Pleural Other', 'Fracture','Support Devices']]
train_df = train_df.fillna(0)

# +
# Counting number of uncertain findings in each pathology

print(f"Cardiomegaly -1: {len(train_df[train_df['Cardiomegaly'] == -1])}")
print(f"Edema -1: {len(train_df[train_df['Edema'] == -1])}")
print(f"Consolodation -1: {len(train_df[train_df['Consolidation'] == -1])}")
print(f"Atelectasis -1: {len(train_df[train_df['Atelectasis'] == -1])}")
print(f"Pleural -1: {len(train_df[train_df['Pleural Effusion'] == -1])}")

# +
# UIgnore - Ignore rows where there is at least one uncertain reading
mask = (train_df != -1.0).all(axis=1)

# Apply the mask to filter the DataFrame
train_df_filtered = train_df[mask]

# Keeping the uncertain labels
train_df_uncertain = train_df[mask == False]
# -

print(len(train_df_filtered))
print(len(train_df_uncertain))

valid_df = valid_df[['Path','Atelectasis','Cardiomegaly','Consolidation','Edema','Pleural Effusion',
                     'No Finding','Enlarged Cardiomediastinum', 'Lung Opacity','Lung Lesion','Pneumonia',
                     'Pneumothorax', 'Pleural Other', 'Fracture','Support Devices']]

# +
#Load image paths

train_image_paths = [source_path + '/' + path for path in train_df_filtered['Path']]
train_image_uncertain_paths = [source_path + '/' + path for path in train_df_uncertain['Path']]
valid_image_paths = [source_path + '/' + path for path in valid_df['Path']]

# Create TensorFlow tensors from image paths
train_image_paths = tf.constant(train_image_paths)
train_image_uncertain_paths = tf.constant(train_image_uncertain_paths)
valid_image_paths = tf.constant(valid_image_paths)
# -

print(len(train_image_paths))
print(len(train_image_uncertain_paths))

# +
# Dropping path as no longer required

train_df = train_df_filtered.drop(['Path'], axis=1)
train_df_uncertain = train_df_uncertain.drop(['Path'], axis=1)
valid_df = valid_df.drop(['Path'], axis=1)

# +
# Convert to array

train_labels = np.array(train_df)
train_labels_uncertain = np.array(train_df_uncertain)
valid_labels = np.array(valid_df)
# -

# # Load and Resize Image

# +
from tqdm import tqdm
from keras.preprocessing import image

#training images preprocessing
SIZE = 320

# Define a custom preprocessing function
def preprocess_image(image_path, label):
    # Read the image file
    image = tf.io.read_file(image_path)
    # Decode the image from bytes to a tensor
    image = tf.image.decode_jpeg(image, channels=3)
#     # Resize the image to a fixed size
#     image = tf.image.resize(image, [SIZE, SIZE])
    # Normalize pixel values to be in the range [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


# -

# # Prepare the data pipeline by setting batch size & buffer size using tf.data 

# Create TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
valid_ds = tf.data.Dataset.from_tensor_slices((valid_image_paths, valid_labels))
train_ds_uncertain = tf.data.Dataset.from_tensor_slices((train_image_uncertain_paths, train_labels_uncertain))

# +
batch_size = 16
AUTOTUNE = tf.data.AUTOTUNE

# Apply preprocessing function to the datasets
train_ds = train_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
train_ds_uncertain = train_ds_uncertain.map(preprocess_image, num_parallel_calls=AUTOTUNE)
# -

# # Visualize Sample Image

# +
# Plot a sample of 10 original images
fig, axes = plt.subplots(1, 10, figsize=(16, 15))  # Adjust the figsize as needed
axes = axes.flatten()

for i, (image, label) in enumerate(train_ds.take(10)):
    ax = axes[i]
    ax.imshow(image.numpy())  # Select the first image from the batch
    ax.set_axis_off()

plt.tight_layout()
plt.show()
# -

# # Augementation

# +
import tensorflow_datasets as tfds
from tensorflow.keras import layers

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(
        height_factor=(-0.05, -0.15),
        width_factor=(-0.05, -0.15)),
    #layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
])


# +
def prepare(ds, shuffle=False, augment=False):
    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.batch(batch_size)
    ds = ds.cache()

#     if augment:
#         ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
#                     num_parallel_calls=AUTOTUNE)

    return ds.prefetch(buffer_size=AUTOTUNE)


# -

train_ds = prepare(train_ds, shuffle=True, augment=True)
train_ds_uncertain = prepare(train_ds_uncertain, shuffle=True, augment=True)
valid_ds = prepare(valid_ds)


# # Visualize Augmented Image

# +
# Define a function to plot sample images
def plot_sample_images(dataset, num_samples=10):
    # Create an iterator for the dataset
    iterator = iter(dataset)

    # Get the next batch of images and labels
    sample_images, sample_labels = next(iterator)

    # Plot the sample images
    fig, axes = plt.subplots(1, num_samples, figsize=(16, 15))
    axes = axes.flatten()

    for i in range(num_samples):
        img = sample_images[i]
        ax = axes[i]
        ax.imshow(img.numpy())  # Convert TensorFlow tensor to NumPy array for plotting
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()

# Visualize sample images from the training dataset
plot_sample_images(train_ds, num_samples=10)


# -

# # Build the Model

# +
# Horovod training

def create_model():
    model = tf.keras.models.Sequential()
    pre_trained_model = tf.keras.applications.densenet.DenseNet121(
        include_top=False,
        weights=None,
        input_shape=(320, 320, 3)
    )

    model.add(pre_trained_model)
    model.add(GlobalAveragePooling2D(input_shape=(1024, 1, 1)))
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=14, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001 * hvd.size(), beta_1=0.9, beta_2=0.999)  # adjust learning rate based on number of GPUs
    opt = hvd.DistributedOptimizer(opt)
    
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy', tf.keras.metrics.AUC(multi_label=True, num_labels=14)])

    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999),
    #               loss='binary_crossentropy',
    #               metrics=['binary_accuracy', tf.keras.metrics.AUC(multi_label=True, num_labels=5)])


    return model


# +
# def create_model():
#     model = tf.keras.models.Sequential()
#     pre_trained_model = tf.keras.applications.densenet.DenseNet121(
#         include_top=False,
#         weights='imagenet',
#         input_shape=(320, 320, 3)
#     )

# #     for layer in pre_trained_model.layers:
# #         layer.trainable = False

#     model.add(pre_trained_model)
#     model.add(GlobalAveragePooling2D(input_shape=(1024, 1, 1)))
#     model.add(Dense(2048, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.2))
#     model.add(Dense(512, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.2))
#     model.add(tf.keras.layers.Dense(units=14, activation='sigmoid'))

#     #     model.add(tf.keras.layers.Flatten())
#     #     model.add(tf.keras.layers.Dense(units = 512, activation = 'relu'))
#     #     model.add(tf.keras.layers.Dense(units = 5, activation = 'sigmoid'))

#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999),
#                   loss='binary_crossentropy',
#                   metrics=['binary_accuracy', tf.keras.metrics.AUC(multi_label=True, num_labels=14)])

#     return model
# -

# # Train the Model

# +
# Horovod training

class CustomSaveModel(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, save_interval):
        super(CustomSaveModel, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        self.iteration = 0

    def on_batch_end(self, batch, logs=None):
        self.iteration += 1
        if self.iteration % self.save_interval == 0:
            model_checkpoint = os.path.join(self.checkpoint_dir, f"model_checkpoint_{self.iteration}.h5")
            self.model.save(model_checkpoint)
            if hvd.rank() == 0:
                print(f"Saved checkpoint at batch {self.iteration} to {model_checkpoint}")

def create_callbacks(run):
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback()
    ]
    
    if hvd.rank() == 0:
        # TensorBoard callback
        log_dir = "logs/fit/" + str(run) + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback)
        
        # Model Checkpoint callback using CustomSaveModel
        checkpoint_dir = f"logs/fit/uself/run_{run}"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Save every 100 iterations
        save_freq_in_batches = 4800
        custom_checkpoint_callback = CustomSaveModel(checkpoint_dir, save_freq_in_batches)
        callbacks.append(custom_checkpoint_callback)

    return callbacks


# +
# import time

# def create_callbacks(run_num):
#     # Modify the log_dir to include 'uignore' and the run number.
#     log_dir = "logs/fit/uself/run_" + str(run_num) + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
#     # Modify the checkpoint_dir to include the run number.
#     checkpoint_dir = f"logs/fit/uself/run_{run_num}"

#     # Ensure the directory exists
#     if not os.path.exists(checkpoint_dir):
#         os.makedirs(checkpoint_dir)

#     # SaveCheckpointCallback class definition
#     class SaveCheckpointCallback(tf.keras.callbacks.Callback):
#         def __init__(self, checkpoint_dir, save_interval):
#             super(SaveCheckpointCallback, self).__init__()
#             self.checkpoint_dir = checkpoint_dir
#             self.save_interval = save_interval
#             self.iteration = 0

#         def on_batch_end(self, batch, logs=None):
#             self.iteration += 1
#             if self.iteration % self.save_interval == 0:
#                 model_checkpoint = os.path.join(self.checkpoint_dir, f"model_checkpoint_{self.iteration}.h5")
#                 self.model.save(model_checkpoint)
#                 print(f"Saved checkpoint at iteration {self.iteration} to {model_checkpoint}")

#     save_interval = 4800 
#     checkpoint_callback = SaveCheckpointCallback(checkpoint_dir, save_interval)
    
#     return [checkpoint_callback, tensorboard_callback]

# +
# Horovod training

trained_models = []
def train(num_runs, train_ds, valid_ds):
    for run in range(num_runs):
        # Only rank 0 will print this log
        if hvd.rank() == 0:
            print(f"Run {run + 1} of {num_runs}")

        # Clear previous session to ensure a fresh start for each run
        tf.keras.backend.clear_session()

        model = create_model()
        callbacks = create_callbacks(run+1)

        start = time.time()
        history = model.fit(train_ds, epochs=4, validation_data=valid_ds, batch_size=batch_size, callbacks=callbacks, verbose=2 if hvd.rank() == 0 else 0)
        
        # Only rank 0 will print this log
        if hvd.rank() == 0:
            print("Total time for run", run + 1, ": ", time.time() - start, "seconds")

        trained_models.append(model)

    return trained_models

# Define the number of runs
num_runs = 1
training = train(num_runs, train_ds, valid_ds)

# +
# trained_models = []

# def train(num_runs, train_ds, valid_ds):

#     for run in range(num_runs):
#         print(f"Run {run + 1} of {num_runs}")

#         # Clear previous session to ensure a fresh start for each run
#         tf.keras.backend.clear_session()

#         model = create_model()
#         callbacks = create_callbacks(run+1)

#         start = time.time()
#         history = model.fit(train_ds, epochs=4, validation_data=valid_ds, batch_size=batch_size, callbacks=callbacks, verbose=2)
#         print("Total time for run", run + 1, ": ", time.time() - start, "seconds")
        
#         trained_models.append(model)

#     return trained_models

# # Define the number of runs
# num_runs = 1
# training = train(num_runs, train_ds, valid_ds)

# +
# # %tensorboard --logdir logs --port 8885
# -

# # Evaluate the Model with Uncertain Labels

# +
all_predictions_uncertain = []
all_actual_labels_uncertain = []

# Iterate over the dataset and collect predictions for individual images
for model in trained_models:
    predictions = model.predict(train_ds_uncertain, batch_size = 16)  # Get predictions for the current batch of images
    all_predictions_uncertain.extend(predictions)


# +
all_predictions_uncertain = []

# Get the first model
first_model = trained_models[0]

# Get predictions for the uncertain dataset using only the first model
predictions = first_model.predict(train_ds_uncertain, batch_size=16)

# Extend the all_predictions_uncertain list with the predictions from the first model
all_predictions_uncertain.extend(predictions)

# -

all_predictions_uncertain = np.array(all_predictions_uncertain)
print(all_predictions_uncertain.shape)
print(train_labels_uncertain.shape)

# # Re-labeling uncertain labels

# +
# Re-labeling each of the uncertainty labels without replacing instances of 0 and 1

mask = train_labels_uncertain == -1
print(mask.shape)
new_prediction = np.where(mask,all_predictions_uncertain,train_labels_uncertain)

print("Actual vs. Prediction for the first 10 images:")
for i in range(2):
    prediction_2 = new_prediction[i]
    actual_2= np.array(train_labels_uncertain[i]).astype(int)
    print(f"Actual Labels: {actual_2}")
    print(f"New Predictions  : {prediction_2}")
# -

# # Retraining the Model including the re-labeled data

# +
# Combine the image paths
combined_paths = []
combined_paths.extend(train_image_paths)
combined_paths.extend(train_image_uncertain_paths)

combined_labels = []
combined_labels.extend(train_labels)
combined_labels.extend(new_prediction)
combined_labels = np.array(combined_labels)
# -

print(len(combined_paths))
print(len(combined_labels))

# Create a tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices((combined_paths, combined_labels))
train_ds = train_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
train_ds = prepare(train_ds, shuffle=True, augment=True)

# Define the number of runs
trained_models = []
num_runs2 = 3
training = train(num_runs2, train_ds, valid_ds)

# %tensorboard --logdir logs --port 8885

# # Model Evaluate on Checkpoint

len(trained_models)

# +
# Initialize a list to store checkpoint paths for each run
checkpoint_paths_list = []

for run in range(num_runs):
    checkpoint_paths = []  # Store checkpoint paths for the current model

    # Collect checkpoint paths
    for iteration in range(4800, 52801, 4800):  # Modify this range according to your save_interval and number of checkpoints
        checkpoint_path = f"logs/fit/uself/run_{run + 1}/model_checkpoint_{iteration}.h5"
        checkpoint_paths.append(checkpoint_path)

    checkpoint_paths_list.append(checkpoint_paths)


# +
# # Initialize a list to store checkpoint paths for each run
# checkpoint_paths_list = []

# for model in trained_models:
#     checkpoint_paths = []  # Store checkpoint paths for the current model

#     # Collect checkpoint paths
#     for iteration in range(4800, 52801, 4800):  # Modify this range according to your save_interval
#         checkpoint_path = f"logs/fit/uselftrained/model_checkpoint_{iteration}.h5"
#         checkpoint_paths.append(checkpoint_path)

#     checkpoint_paths_list.append(checkpoint_paths)

# +
# Initialize a list to store predictions for each checkpoint
all_predictions = []

# Iterate through the collected checkpoint paths and corresponding trained model
for model, checkpoint_paths in zip(trained_models, checkpoint_paths_list):
    predictions = []  # Store predictions for the current run

    # Load each checkpoint and predict on the validation set
    for checkpoint_path in checkpoint_paths:
        model.load_weights(checkpoint_path)

        # Predict on the validation set
        checkpoint_predictions = model.predict(valid_ds)
        predictions.append(checkpoint_predictions)

    all_predictions.append(predictions)

# +
# # Initialize a list to store predictions for each checkpoint
# all_predictions = []

# # Iterate through the collected checkpoint paths
# for checkpoint_paths in checkpoint_paths_list:
#     predictions = []  # Store predictions for the current run

#     # Load each checkpoint and predict on the validation set
#     for checkpoint_path in checkpoint_paths:
#         model.load_weights(checkpoint_path)

#         # Predict on the validation set
#         checkpoint_predictions = model.predict(valid_ds)
#         predictions.append(checkpoint_predictions)

#     all_predictions.append(predictions)

# +
from sklearn.metrics import roc_auc_score

average_auroc_list = []
num_pathologies = 5
iteration_auroc = []

for checkpoint_predictions in all_predictions:
    
    for checkpoint_index, checkpoint_prediction in enumerate(checkpoint_predictions):
        checkpoint_auroc_scores = []  # Store AUROC scores for the current model
        
        for pathology_index in range(num_pathologies):
            true_labels = valid_labels[:, pathology_index]
            auroc = roc_auc_score(true_labels, checkpoint_prediction[:, pathology_index])
            checkpoint_auroc_scores.append(auroc)
        
        iteration_auroc.append(checkpoint_auroc_scores)
        
# Calculate the average AUROC for this checkpoint
iteration_auroc = np.array(iteration_auroc)
average_auroc = np.mean(iteration_auroc, axis = 1)

# Calculate the indices that would sort the average AUROC list in descending order
sorted_indices = np.argsort(average_auroc)[::-1]

# Get the top 30 indices
top_30_indices = sorted_indices[:20]

# Initialize a list to store the corresponding checkpoint_auroc_scores
best_checkpoint_auroc_scores = []

# Extract the checkpoint_auroc_scores for the best 30 averages
for index in top_30_indices:
    best_checkpoint_auroc_scores.append(iteration_auroc[index])

best_checkpoint_auroc_scores = np.array(best_checkpoint_auroc_scores)

auroc_pathology = np.mean(best_checkpoint_auroc_scores, axis = 0)
print(f"auroc_pathology: {auroc_pathology}")

# Overall AUROC
overall_ave = np.mean(best_checkpoint_auroc_scores)
print(f"overall_ave_AUROC: {overall_ave}")

# +
# Calculate standard deviation (confidence intervals) for each pathology
std_dev_pathology = np.std(best_checkpoint_auroc_scores, axis=0)
print(f"Standard Deviation (confidence intervals) for each pathology: {std_dev_pathology}")

confidence_intervals = [(auroc - 1.96 * std, auroc + 1.96 * std) for auroc, std in zip(auroc_pathology, std_dev_pathology)]
print("Confidence Intervals for each pathology:")
for i, (lower_bound, upper_bound) in enumerate(confidence_intervals):
    print(f"Pathology {i + 1}: ({lower_bound:.4f}, {upper_bound:.4f})")
