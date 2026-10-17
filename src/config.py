"""Shared experimental settings.

Every architecture is trained with these values unchanged so differences in
loss, perplexity, accuracy, and generated text can be attributed to the
recurrent cell rather than to a different optimisation setup.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
PLOT_DIR = RESULTS_DIR / "plots"
CHECKPOINT_DIR = ROOT / "checkpoints"

BOOKS = [
    {
        "key": "jungle_book",
        "title": "The Jungle Book",
        "year": 1894,
        "filename": "the_jungle_book.txt",
        "url": "https://www.gutenberg.org/ebooks/236",
        "plain_text_url": "https://www.gutenberg.org/cache/epub/236/pg236.txt",
    },
    {
        "key": "second_jungle_book",
        "title": "The Second Jungle Book",
        "year": 1895,
        "filename": "the_second_jungle_book.txt",
        "url": "https://www.gutenberg.org/ebooks/1937",
        "plain_text_url": "https://www.gutenberg.org/cache/epub/1937/pg1937.txt",
    },
    {
        "key": "just_so_stories",
        "title": "Just So Stories",
        "year": 1902,
        "filename": "just_so_stories.txt",
        "url": "https://www.gutenberg.org/ebooks/2781",
        "plain_text_url": "https://www.gutenberg.org/cache/epub/2781/pg2781.txt",
    },
]

SEED = 42
SEQ_LEN = 20
STRIDE = 2
MIN_FREQ = 2
EMBED_DIM = 128
HIDDEN_DIM = 256
DROPOUT = 0.2
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
EPOCHS = 10
TRAIN_FRACTION = 0.85

ARCHITECTURES = ["rnn", "birnn", "lstm", "gru"]
ARCH_LABELS = {
    "rnn": "Simple RNN",
    "birnn": "Bidirectional RNN",
    "lstm": "LSTM",
    "gru": "GRU",
}

TEMPERATURES = [0.5, 1.0, 1.5]
GENERATED_WORDS = 70

# Prompts are checked against the training vocabulary at runtime.
# Any prompt that would require an unknown token is dropped.
CANDIDATE_SEEDS = [
    "in the sea once upon a time",
    "now chil the kite brings home the night",
    "this is the hour of pride and power",
    "o my best beloved there was a whale",
    "in the high and far off times",
    "the cat that walked by himself",
    "hear and attend and listen",
]
