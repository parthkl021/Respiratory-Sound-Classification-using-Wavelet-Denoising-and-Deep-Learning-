import wave
import numpy as np
import pywt
import glob
from scipy.signal import resample
import pandas as pd
import os
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from keras.layers import Flatten, Dense, Dropout, Bidirectional, LSTM, TimeDistributed
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import EarlyStopping
import soundfile as sf
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix


def process_audio_and_labels(wav_file_path, label_file_path, max_length):
    # Load and process the audio data
    audio_data, samplerate = sf.read(wav_file_path)
    if audio_data.dtype != np.int16:
        if np.issubdtype(audio_data.dtype, np.integer):
            max_val = np.iinfo(audio_data.dtype).max
            audio_data = (audio_data.astype(np.float32) / max_val).clip(-1, 1) * np.iinfo(np.int16).max
            audio_data = audio_data.astype(np.int16)
        elif np.issubdtype(audio_data.dtype, np.floating):
            audio_data = (audio_data.clip(-1, 1) * np.iinfo(np.int16).max).astype(np.int16)
        else:
            raise ValueError(f"Unsupported data type {audio_data.dtype}")
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_data = audio_data.mean(axis=1)

    new_n_samples = int(len(audio_data) * 4000 / samplerate)
    audio_data = resample(audio_data, new_n_samples)
    frame_rate = 4000

    # Apply your denoising process here
    duration = len(audio_data) / frame_rate

    fft_data = np.fft.fft(audio_data)
    frequency = np.fft.fftfreq(len(audio_data), d=1/frame_rate)
    fft_data_filtered = fft_data.copy()
    fft_data_filtered[(np.abs(frequency) < 15) | (np.abs(frequency) > 1700)] = 0 
    filtered_audio_data = np.fft.ifft(fft_data_filtered).real

    wavelet_name = 'coif2'
    n_layers = 9
    coeffs = pywt.wavedec(filtered_audio_data, wavelet_name, level=n_layers)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2*np.log(len(filtered_audio_data)))
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], uthresh, mode='soft')

    denoised_signal = pywt.waverec(coeffs, wavelet_name)
    
    sample_rate = 4000
    audio_data = (audio_data - np.mean(audio_data)) / np.std(audio_data)
    denoised_signal = (denoised_signal - np.mean(denoised_signal)) / np.std(denoised_signal)

    # Initialize the new signal and new labels
    new_signal = []
    new_labels = []

    # The duration of the 'None' signals in seconds
    none_duration = 0.2
    sample_rate = 4000
    # The 'None' signal (a sequence of zeros)
    none_signal = [0] * int(none_duration * sample_rate)

    with open(label_file_path, 'r') as label_file:
        labels = label_file.readlines()

    # Check if both 'I' and 'E' labels are present
    label_types = [label.split()[0] for label in labels]
    if 'I' not in label_types or 'E' not in label_types:
        return None, None  # Return None if either 'I' or 'E' is missing

    # Iterate over the labels
    for label in labels:
        label_parts = label.split()
        label_type = label_parts[0]
        start_time = label_parts[1]
        end_time = label_parts[2]

        start_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(start_time.split(':'))))
        end_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(end_time.split(':'))))

        # If the label is either 'I' or 'E', add the corresponding part of the signal to the new signal
        if label_type in ['I', 'E']:
            start_index = int(start_seconds * sample_rate)
            end_index = int(end_seconds * sample_rate)
            signal_segment = denoised_signal[start_index:end_index]
            new_signal.extend(signal_segment)
            new_labels.extend([label_type] * len(signal_segment))  # store label for each sample in signal_segment

            if label != labels[-1]:
                new_signal.extend(none_signal)
                new_labels.extend(['None'] * len(none_signal))  # 'None' label for none_signal

    # Convert the new signal and labels to numpy arrays
    new_signal = np.array(new_signal)
    new_labels = np.array(new_labels)

    # Check if signal is too short, and pad if necessary
    if len(new_signal) < max_length:
        padding = [0] * (max_length - len(new_signal))
        new_signal = np.concatenate([new_signal, padding])
        new_labels = np.concatenate([new_labels, ['None'] * len(padding)])  # pad labels with 'None'
    # If signal is too long, trim it
    elif len(new_signal) > max_length:
        new_signal = new_signal[:max_length]
        new_labels = new_labels[:max_length]  # trim labels

    return new_signal, new_labels


# Function to process all files
def process_all_files(audio_dir, label_dir, max_length):
    X_train = []
    y_train = []
    label_mapping = {'I': 0, 'E': 1, 'None': 2}  # map labels to numbers
    
    for file_name in os.listdir(audio_dir):
        if file_name.endswith(".wav"):
            # Create paths to the audio file and its corresponding label file
            wav_file_path = os.path.join(audio_dir, file_name)
            label_file_path = os.path.join(label_dir, file_name.replace(".wav", "_label.txt"))
            
            # Process the audio and labels
            new_signal, new_labels = process_audio_and_labels(wav_file_path, label_file_path, max_length)

            if new_signal is None or new_labels is None:
                continue

            # Append the new signal and labels to the training data
            X_train.append(new_signal)
            
            # Here, we're encoding 'I' as 0, 'E' as 1 and 'None' as 2
            labels_encoded = [label_mapping[label] for label in new_labels]
            y_train.append(labels_encoded)

    # Convert the training data to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train



directory_path = '/home/iiitd/gautami/Parth/HF_Lung_V1-master/train/train/'
labels_directory = '/home/iiitd/gautami/Parth/HF_Lung_V1-master/train/train/'
None_label = 2
sample_rate = 4000
total_time = 15  
all_labels = {'I', 'Rhonchi', 'E', 'Wheeze', 'Stridor', 'D'}
target_labels = {'I', 'E'}
label_mapping = {'I': 0, 'E': 1, 'None': 2}


# Function to process all files

X_train, y_train = process_all_files(directory_path, labels_directory, max_length = 9 * 4000)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
y_train_one_hot = to_categorical(y_train, num_classes=3)
input_shape = (X_train.shape[1], 1)
X_train, X_val, y_train_one_hot, y_val_one_hot = train_test_split(X_train, y_train_one_hot, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_train.shape[1])
# Define your model
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv1D(96, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.3))
model.add(TimeDistributed(Dense(128, activation='relu')))
model.add(Dropout(0.3))
model.add(TimeDistributed(Dense(3, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy', 'Precision', 'Recall'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')
model.fit(X_train, y_train_one_hot, batch_size=2, epochs=10, validation_split=0.2, callbacks=[early_stopping], verbose=1)
directory_path_for_test = '/home/iiitd/gautami/Parth/HF_Lung_V1-master/test/'
labels_directory_test = '/home/iiitd/gautami/Parth/HF_Lung_V1-master/test/'
# Process the test data
X_test, y_test = process_all_files(directory_path_for_test, labels_directory_test, max_length = 9 * 4000)

# print(X_test.shape)
X_test = np.expand_dims(X_test, axis=-1)  # Add the channel dimension

# One hot encode the labels
y_test_one_hot = to_categorical(y_test, num_classes=3)

# Evaluate the model
evaluation = model.evaluate(X_test, y_test_one_hot, verbose=1)

print('Loss:', evaluation[0])
print('Accuracy:', evaluation[1])
print('Precision:', evaluation[2])
print('Recall:', evaluation[3])

# Get the model's predictions
y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=-1)

y_test_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()

print(classification_report(y_test_flat, y_pred_flat, target_names=['I', 'E', 'None']))
print(confusion_matrix(y_test_flat, y_pred_flat))


# model = Sequential()
# model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
# model.add(BatchNormalization())
# model.add(Conv1D(96, kernel_size=3, activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Bidirectional(LSTM(128, return_sequences=True)))
# model.add(Dropout(0.4))
# model.add(TimeDistributed(Dense(128, activation='relu')))
# model.add(Dropout(0.4))
# model.add(TimeDistributed(Dense(3, activation='softmax')))
# model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy', 'Precision', 'Recall'])
# early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')
#               precision    recall  f1-score   support

#            I       0.62      0.72      0.66  10613966
#            E       0.60      0.50      0.55   9643054
#         None       0.99      0.98      0.99  13762980

#     accuracy                           0.76  34020000
#    macro avg       0.74      0.73      0.73  34020000
# weighted avg       0.76      0.76      0.76  34020000

# [[ 7600236  2980690    33040]
#  [ 4707481  4862758    72815]
#  [    3668   277195 13482117]]


# model = Sequential()
# model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
# model.add(BatchNormalization())
# model.add(Conv1D(96, kernel_size=3, activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Bidirectional(LSTM(128, return_sequences=True)))
# model.add(Dropout(0.4))
# model.add(TimeDistributed(Dense(128, activation='relu')))
# model.add(Dropout(0.4))
# model.add(TimeDistributed(Dense(3, activation='softmax')))
# model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy', 'Precision', 'Recall'])
# early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')
# model.fit(X_train, y_train_one_hot, batch_size=6, epochs=3, validation_split=0.2, callbacks=[early_stopping], verbose=1)
#              precision    recall  f1-score   support

#            I       0.63      0.83      0.72  10613966
#            E       0.72      0.46      0.56   9643054
#         None       0.98      1.00      0.99  13762980

#     accuracy                           0.79  34020000
#    macro avg       0.78      0.76      0.76  34020000
# weighted avg       0.80      0.79      0.78  34020000

# [[ 8768970  1758178    86818]
#  [ 5064554  4451717   126783]
#  [    5659     7879 13749442]]

