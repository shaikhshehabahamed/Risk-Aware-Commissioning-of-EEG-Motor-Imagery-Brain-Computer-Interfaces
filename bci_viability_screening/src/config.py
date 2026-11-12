from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent   # resting_bci_screening/
DATA_RAW = ROOT.parent / "resting_bci_screening" / "eeg-motor-movementimagery-dataset-1.0.0"
RESULTS  = ROOT / "results"
FIGURES  = ROOT / "figures"
TABLES   = ROOT / "tables"

for _d in (RESULTS, FIGURES, TABLES):
    _d.mkdir(exist_ok=True)

# ── Recording constants ────────────────────────────────────────────────────────
SFREQ      = 160.0   # Hz — nominal sampling rate for included subjects
N_CHANNELS = 64      # expected EEG channel count

# Runs used in this study
RUN_BASELINE     = "R02"
RUN_BASELINE_EO  = "R01"           # eyes-open baseline for alpha reactivity
TASK2_RUNS       = ["R04", "R08", "R12"]   # Task 2: left vs right hand MI (primary)
TASK4_RUNS       = ["R06", "R10", "R14"]   # Task 4: bilateral MI (kept for reference)

# ── Subject inclusion thresholds ───────────────────────────────────────────────
MIN_BASELINE_DURATION = 50.0   # seconds of usable R02 signal (after first 1 s skip)
MIN_EPOCHS_PER_CLASS  = 5      # required per class per task run, after rejection

# ── Task preprocessing / viability label ──────────────────────────────────────
TASK_BANDPASS     = (8.0, 30.0)   # Hz, 4th-order Butterworth zero-phase
EPOCH_TMIN        = 0.5           # s after cue onset
EPOCH_TMAX        = 4.0           # s after cue onset
REJECT_THRESH_V   = 400e-6        # 400 µV peak-to-peak amplitude rejection

# BCI2000 annotation labels → binary class mapping
# Task 2: T1 = left fist MI (class 0), T2 = right fist MI (class 1)
# Task 4: T1 = both fists MI (class 0), T2 = both feet MI (class 1)
TASK_EVENT_LABELS = {"T1": 0, "T2": 1}   # same annotation names for both tasks

# Backwards-compatible aliases
TASK4_BANDPASS    = TASK_BANDPASS
TASK4_EVENT_LABELS = TASK_EVENT_LABELS

# FBCSP sub-bands and CSP components
FBCSP_SUBBANDS   = [(8, 12), (12, 16), (16, 20), (20, 30)]
CSP_N_COMPONENTS = 4              # 4 CSP components per sub-band → 16 features total

VIABILITY_THRESHOLD = 0.60        # Fold-3 prospective accuracy >= 60% -> viable

# ── Resting-state preprocessing ───────────────────────────────────────────────
REST_BANDPASS   = (1.0, 40.0)   # Hz, wider than task pipeline (microstates need slow freqs)
REST_TRIM_START = 1.0            # skip first 1 s of the recording
REST_DURATION   = 60.0           # seconds of signal to use

# ── Microstate parameters ─────────────────────────────────────────────────────
N_MICROSTATES         = 4
MICROSTATE_N_RUNS     = 50      # random initialisations for global k-means
SMOOTHING_MIN_SAMPLES = 5       # segments shorter than this (< 30 ms) are reassigned

# ── Spectral features — channels ───────────────────────────────────────────────
# Primary motor channels (C3/Cz/C4) + centroparietal (CP3/CPz/CP4)
SPECTRAL_CHANNELS_PRIMARY = ["C3", "Cz", "C4"]
SPECTRAL_CHANNELS_CP      = ["Cp3", "Cpz", "Cp4"]   # dot-stripped BCI2000 names
SPECTRAL_CHANNELS = SPECTRAL_CHANNELS_PRIMARY + SPECTRAL_CHANNELS_CP  # 6 channels total

# Lateralisation and coherence channel pair (dominant hand = right, motor = C4 left)
LATERALISATION_PAIR = ("C3", "C4")    # ipsi vs contralateral to right hand
CP_LATERALISATION_PAIR = ("Cp3", "Cp4")

SPECTRAL_BANDS = {
    "delta": (1.0,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 12.0),
    "smr":   (12.0, 15.0),
    "beta":  (15.0, 30.0),
    "gamma": (30.0, 45.0),
}

ALPHA_PEAK_RANGE = (8.0, 13.0)  # Hz, for individual alpha peak detection

# Welch PSD settings
WELCH_WINDOW_S  = 2.0   # 2-second Hann window = 320 samples at 160 Hz
WELCH_OVERLAP   = 0.5   # 50% overlap
