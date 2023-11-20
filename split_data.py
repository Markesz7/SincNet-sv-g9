import os
import random
import numpy as np

from pydub import AudioSegment

# Parameters (Change these depending on your needs) TODO: Maybe it's nicer to get these as command line arguments
input_dir = "/work/courses/T/S/89/5150/general/data/LibriSpeech/dev-clean"
output_train_dir = "data/training_data"
output_test_dir = "data/testing_data"
split_ratio = 0.7 # Split the data for training and test data with split_ratio
randomise_data = False # If false, training and test data will always be the same.

def list_to_scp(input_list, output_file):
    with open(output_file, 'w') as file:
        for input in input_list:
            file.write("%s\n" % input)

def convert_flac_to_wav(input_flac, output_wav):
    audio = AudioSegment.from_file(input_flac, format="flac")
    audio.export(output_wav, format="wav")



# Keep the labels and scp file paths for exports later
label_dict = {}
scp_list_training = []
scp_list_test = []

# Go through all the speakers
speakers = os.listdir(input_dir)
for speaker in speakers:
    speaker_path = os.path.join(input_dir, speaker)
    if os.path.isdir(speaker_path):
        chapters = os.listdir(speaker_path)

        # Shuffle chapters to randomly split into train and test if it's enabled
        if randomise_data:
            random.shuffle(chapters)

        # Create speaker output directories if they don't exist
        output_train_speaker_dir = os.path.join(output_train_dir, speaker)
        output_test_speaker_dir = os.path.join(output_test_dir, speaker)
        os.makedirs(output_train_speaker_dir, exist_ok=True)
        os.makedirs(output_test_speaker_dir, exist_ok=True)

        # Counting all the flac_files, so we know where to split up the data (Later this could be optimized better)
        number_of_flac_files = 0
        for chapter in chapters:
            chapter_path = os.path.join(speaker_path, chapter)
            number_of_flac_files += len(os.listdir(chapter_path))
        
        i = 0    # Keep the counter for the flac_files, so we know when to switch for the test data in the for loop
        for chapter in chapters:
            chapter_path = os.path.join(speaker_path, chapter)

            split_number = number_of_flac_files * split_ratio

            flac_files = os.listdir(chapter_path)

            # Shuffle flac_files to randomly split into train and test if it's enabled    
            if randomise_data:
                random.shuffle(flac_files)

            # Convert and copy each FLAC file to WAV to training/test data depending on the counter
            for flac_file in flac_files:
                if flac_file.endswith(".flac"):
                    flac_path = os.path.join(chapter_path, flac_file)

                    # If counter is lower than the split, it goes into training data, else it goes into the test data
                    if i < split_number: 
                        wav_file = os.path.join(output_train_speaker_dir, os.path.splitext(flac_file)[0] + ".wav")
                        scp_list_training.append(wav_file)
                    else:
                        wav_file = os.path.join(output_test_speaker_dir, os.path.splitext(flac_file)[0] + ".wav")
                        scp_list_test.append(wav_file)
                    
                    label_dict[wav_file] = speaker
                    convert_flac_to_wav(flac_path, wav_file)
                i += 1
                        

# Export the scp files
scp_list_all = scp_list_training + scp_list_test
list_to_scp(scp_list_training, "dev_clean_training.scp")
list_to_scp(scp_list_test, "dev_clean_test.scp")
list_to_scp(scp_list_all, "dev_clean_all.scp")

# Export the numpy file with the labels
np.save('dev_clean_labels.npy', label_dict)