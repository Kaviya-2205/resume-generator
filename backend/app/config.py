# ResumeGenAIBackend/app/config.py

import os

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(_file_)))

DATASET_PATH = os.path.join(BASE_DIR, "dataset.json")
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "model.pth")

# Model parameters
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
MAX_SEQ_LEN = 100
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.003
MAX_VOCAB_SIZE = 5000