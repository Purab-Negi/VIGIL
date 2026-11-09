"""Central configuration for the VIGIL pipeline (paths + every hyperparameter the paper leaves open)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- inputs -----------------------------------------------------------------
DOWNLOADS = Path.home() / "Downloads"
AUDIO_ZIP = DOWNLOADS / "Audio DataSet - Anotated.zip"
ANNOTATION_DIR = DOWNLOADS / "VIGIL_Annotation"
ANNOTATIONS_CSV = ANNOTATION_DIR / "annotations.csv"
LOSO_FOLDS_CSV = ANNOTATION_DIR / "loso_folds.csv"
# NOTE: session_metadata_SYNTHETIC_v2.csv is synthetic and is deliberately NOT used anywhere.

# ---- working directories ----------------------------------------------------
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"            # extracted original files, renamed to anonymous utt_id
WAV_DIR = DATA / "wav16k"         # Sec. IV-C: mono 16 kHz PCM WAV, no lossy re-encoding
SPEECH_DIR = DATA / "speech"      # pangram-only, VAD-trimmed speech
FEAT_DIR = DATA / "features"
MODEL_CACHE = ROOT / "pretrained_models"
RESULTS = ROOT / "results"

# ---- signal processing (Sec. IV-C) -------------------------------------------
SAMPLE_RATE = 16000
CLIP_THRESHOLD = 0.999            # |x| >= this counts as a clipped sample
CLIP_FLAG_RATIO = 0.001           # flag if >0.1 % of samples are clipped
LOW_PEAK_DBFS = -30.0             # flag unusually quiet recordings
LOW_SNR_DB = 15.0                 # flag unusually low estimated SNR
VAD_THRESHOLD = 0.5               # silero-vad speech probability threshold
VAD_MIN_SILENCE_MS = 250          # pauses longer than this are removed
VAD_SPEECH_PAD_MS = 30
SEGMENT_S = 3.0                   # fixed-length overlapping segments
HOP_S = 1.5                       # 50 % overlap
MIN_TAIL_S = 1.0                  # add an end-aligned window if >= this much speech is left over

# ---- self-introduction removal (Sec. IV-B) -----------------------------------
WHISPER_MODEL = "small.en"
INTRO_PAD_S = 0.10                # start this much before the first pangram word

# ---- embeddings (Sec. IV-D) ----------------------------------------------------
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
WAVLM_SOURCE = "microsoft/wavlm-base-plus"
WAVLM_LAYER = 6                   # intermediate transformer layer (of 12), mean-pooled over time
FEATURE_SETS = ["ecapa", "wavlm", "egemaps"]

# ---- classification (Sec. IV-F) ------------------------------------------------
PCA_VARIANCE = 0.95               # fixed proportion of variance retained
SVM_GRID = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 1e-3, 1e-2, 1e-1],
}
INNER_FOLDS = 5
GRID_SCORING = "balanced_accuracy"

# ---- evaluation (Sec. IV-G, V-A) -------------------------------------------------
NORMALISATIONS = ["none", "rested_baseline"]   # 'none' is the paper's control condition
RANDOM_SPLIT_FOLDS = 5
RANDOM_SPLIT_REPEATS = 5
BOOTSTRAP_ITERS = 2000
SEED = 42
