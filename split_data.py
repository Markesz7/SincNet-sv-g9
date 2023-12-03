import os
import shutil
import random
import numpy as np

from pydub import AudioSegment

# Parameters (Change these depending on your needs) TODO: Maybe it's nicer to get these as command line arguments
input_dir = "/work/courses/T/S/89/5150/general/data/LibriSpeech/dev-clean"
output_train_dir = "training_data"
output_val_dir = "validation_data"
output_test_dir = "testing_data"
split_ratio1 = 0.7 # Split the data for training and test data with split_ratio
split_ratio2 = 0.85
randomise_data = False # If false, training and test data will always be the same.

def list_to_scp(input_list, output_file):
    with open(output_file, 'w') as file:
        for input in input_list:
            file.write("%s\n" % input)

def convert_flac_to_wav(input_flac, output_wav):
    audio = AudioSegment.from_file(input_flac, format="flac")
    audio.export(output_wav, format="wav")

try:
    shutil.rmtree("data")
    shutil.rmtree("data_lists")
    shutil.rmtree("normalized_data")
    print("Data folders are now deleted, creating new ones!")
except FileNotFoundError:
    print(f"Data folders were not found, creating new ones!")

# Make the folder for the data and change the cwd to it, so it doesn't get included in the scp/npy files.
os.makedirs("data", exist_ok=True)
cwd = os.getcwd()
os.chdir(os.path.join(cwd, "data"))

# Keep the labels and scp file paths for exports later
label_dict = {}
scp_list_training = []
scp_list_val = []
scp_list_test = []

# In the new folders, start the speaker ids from 0, so we can give the sincnet config the right amount of speakers for the softmax layer.
speaker_id = 0

# Go through all the speakers
speakers = os.listdir(input_dir)
number_of_speakers = len(speakers)
for speaker in speakers:
    speaker_path = os.path.join(input_dir, speaker)
    if os.path.isdir(speaker_path):
        chapters = os.listdir(speaker_path)

        # Shuffle chapters to randomly split into train and test if it's enabled
        if randomise_data:
            random.shuffle(chapters)

        # Create speaker output directories if they don't exist
        output_train_speaker_dir = os.path.join(output_train_dir, str(speaker_id))
        output_val_speaker_dir = os.path.join(output_val_dir, str(speaker_id))
        output_test_speaker_dir = os.path.join(output_test_dir, str(speaker_id))
        os.makedirs(output_train_speaker_dir, exist_ok=True)
        os.makedirs(output_val_speaker_dir, exist_ok=True)
        os.makedirs(output_test_speaker_dir, exist_ok=True)

        # Counting all the flac_files, so we know where to split up the data (Later this could be optimized better)
        number_of_flac_files = 0
        for chapter in chapters:
            chapter_path = os.path.join(speaker_path, chapter)
            number_of_flac_files += len(os.listdir(chapter_path))
        
        i = 0    # Keep the counter for the flac_files, so we know when to switch for the test data in the for loop
        for chapter in chapters:
            chapter_path = os.path.join(speaker_path, chapter)

            split_number1 = number_of_flac_files * split_ratio1
            split_number2 = number_of_flac_files * split_ratio2

            flac_files = os.listdir(chapter_path)

            # Shuffle flac_files to randomly split into train and test if it's enabled    
            if randomise_data:
                random.shuffle(flac_files)

            # Convert and copy each FLAC file to WAV to training/test data depending on the counter
            for flac_file in flac_files:
                if flac_file.endswith(".flac"):
                    flac_path = os.path.join(chapter_path, flac_file)

                    # If counter is lower than the split, it goes into training data, else it goes into the test data
                    if i < split_number1: 
                        wav_file = os.path.join(output_train_speaker_dir, os.path.splitext(flac_file)[0] + ".wav")
                        scp_list_training.append(wav_file)
                    elif i < split_number2:
                        wav_file = os.path.join(output_val_speaker_dir, os.path.splitext(flac_file)[0] + ".wav")
                        scp_list_val.append(wav_file)
                    else:
                        wav_file = os.path.join(output_test_speaker_dir, os.path.splitext(flac_file)[0] + ".wav")
                        scp_list_test.append(wav_file)
                    
                    label_dict[wav_file] = speaker_id
                    convert_flac_to_wav(flac_path, wav_file)
                i += 1
    speaker_id += 1
    if(speaker_id % 5 == 0):
        print(f'Preprocessing is {speaker_id / number_of_speakers * 100}% done!')

# Swich back to root directory                        
os.chdir(cwd)

# Export the scp files
os.makedirs("data_lists", exist_ok=True)
scp_list_all = scp_list_training + scp_list_val + scp_list_test
list_to_scp(scp_list_training, "data_lists/dev_clean_training.scp")
list_to_scp(scp_list_val, "data_lists/dev_clean_validation.scp")
list_to_scp(scp_list_test, "data_lists/dev_clean_test.scp")
list_to_scp(scp_list_all, "data_lists/dev_clean_all.scp")

# Export the numpy file with the labels
np.save('data_lists/dev_clean_labels.npy', label_dict)
