# This file has the configurations of the experiments.
import multiprocessing
import os

import torch

USE_CUSTOMIZE_DATASETS = True

DATA_PATH ="../Dataset"
SAVE_NAME_MODEL = ""
TRAIN_DATA_DIR = ""
SAVED_MODEL_PATH = ""

UTT_TEST_FILE_DIRECT="G0112/G0112_0_S0001.wav"

# Number of steps to train.
TRAINING_STEPS = 500

# Set random seed for reproducibility
SEED = 42
SAMPLE_RATE = 16000
if USE_CUSTOMIZE_DATASETS:
    # path of indonesian datasets
    TRAIN_DATA_DIR = os.path.join(f"{DATA_PATH}/datasets/ASR-SIndoDuSC/WAV/train")
    TEST_DATA_DIR = os.path.join(f"{DATA_PATH}/datasets/ASR-SIndoDuSC/WAV/test")
    TESTING_MANUAL_DIR = os.path.join(f"{DATA_PATH}/datasets/Bintang")
    # Path of save model.
    SAVED_MODEL_PATH = os.path.join(
    f"./saved_model/trained/saved_model.bilstm.mean.customize.gpu{str(TRAINING_STEPS)}.pt")
else:
    # Paths of LibriSpeech datasets.
    TRAIN_DATA_DIR = os.path.join(f"{DATA_PATH}/Librispeech/train-clean-100/LibriSpeech/train-clean-100")
    TEST_DATA_DIR = os.path.join(f"{DATA_PATH}/Librispeech/test-clean/LibriSpeech/test-clean")

    # Path of save model.
    SAVED_MODEL_PATH = os.path.join(
    f"./saved_model/trained/saved_model.bilstm.mean.librispeech.gpu{str(TRAINING_STEPS)}.pt")

# Paths of CSV files where the first column is speaker, and the second column is
# utterance file.
# These will allow you to train/evaluate using other datasets than LibriSpeech.
# If given, TRAIN_DATA_DIR and/or TEST_DATA_DIR will be ignored.
TRAIN_DATA_CSV = ""
TEST_DATA_CSV = ""

# Number of MFCCs for librosa.feature.mfcc.
N_MFCC = 40

# Hidden size of LSTM layers.
LSTM_HIDDEN_SIZE = 128

# Number of LSTM layers.
LSTM_NUM_LAYERS = 2

# Whether to use bi-directional LSTM.
BI_LSTM = True

# If false, use last frame of LSTM inference as aggregated output;
# if true, use mean frame of LSTM inference as aggregated output.
FRAME_AGGREGATION_MEAN = True

# If true, we use transformer instead of LSTM.
USE_TRANSFORMER = False

# Dimension of transformer layers.
TRANSFORMER_DIM = 32

# Number of encoder layers for transformer
TRANSFORMER_ENCODER_LAYERS = 2

# Number of heads in transformer layers.
TRANSFORMER_HEADS = 8

# Sequence length of the sliding window for LSTM.
SEQ_LEN = 100  # 3.2 seconds

# Alpha for the triplet loss.
TRIPLET_ALPHA = 0.2

# How many triplets do we train in a single batch.
BATCH_SIZE = 16

# Learning rate.
LEARNING_RATE = 0.000085

# Save a model to disk every these many steps.
SAVE_MODEL_FREQUENCY = 500


# Whether we are going to train with SpecAugment.
SPECAUG_TRAINING = True

# Parameters for SpecAugment training.
SPECAUG_FREQ_MASK_PROB = 0.5
SPECAUG_TIME_MASK_PROB = 0.5
SPECAUG_FREQ_MASK_MAX_WIDTH = N_MFCC // 5
SPECAUG_TIME_MASK_MAX_WIDTH = SEQ_LEN // 5

# Whether to use full sequence inference or sliding window inference.
USE_FULL_SEQUENCE_INFERENCE = False

# Sliding window step for sliding window inference.
SLIDING_WINDOW_STEP = 50  # 1.6 seconds

# Number of triplets to evaluate for computing Equal Error Rate (EER).
# Both the number of positive trials and number of negative trials will be
# equal to this number.
NUM_EVAL_TRIPLETS = 500

# Step of threshold sweeping for computing Equal Error Rate (EER).
EVAL_THRESHOLD_STEP = 0.001

# Number of processes for multi-processing.
NUM_PROCESSES = min(multiprocessing.cpu_count(), BATCH_SIZE)

# Wehther to use GPU or CPU.
DEVICE = torch.device("cuda:0")
