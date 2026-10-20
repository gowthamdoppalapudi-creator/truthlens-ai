"""
TruthLens AI — Central configuration.
Edit RESEARCHER_NAME and APP_SUBTITLE here to customise branding in one place.
"""
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Researcher / Branding
# ──────────────────────────────────────────────────────────────────────────────
RESEARCHER_NAME: str = "D. Venkata Gowtham"  # configured developer name
APP_NAME: str = "TruthLens.ai"
APP_SUBTITLE: str = "Analyze • Predict • Compare"
APP_ICON: str = "🌿"
APP_VERSION: str = "3.0.0"

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR   = ROOT_DIR / "datasets"
REPORTS_DIR = ROOT_DIR / "reports"

# Legacy model paths (original project artefacts — never overwritten silently)
LEGACY_MODEL_PATH = ROOT_DIR / "fake-news-detection-nlp-main" / "final_svc_model.pkl"
if not LEGACY_MODEL_PATH.exists():
    LEGACY_MODEL_PATH = ROOT_DIR / "final_svc_model.pkl"

LEGACY_VEC_PATH = ROOT_DIR / "fake-news-detection-nlp-main" / "tfidf_vectorizer.pkl"
if not LEGACY_VEC_PATH.exists():
    LEGACY_VEC_PATH = ROOT_DIR / "tfidf_vectorizer.pkl"

LEGACY_METRICS = ROOT_DIR / "fake-news-detection-nlp-main" / "metrics.json"
if not LEGACY_METRICS.exists():
    LEGACY_METRICS = ROOT_DIR / "metrics.json"

# Model registry JSON
MODEL_REGISTRY_PATH = MODELS_DIR / "registry.json"

# ──────────────────────────────────────────────────────────────────────────────
# ML defaults
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_TEST_SIZE   = 0.20
DEFAULT_RANDOM_SEED = 42
MIN_SAMPLES_PER_CLASS = 20
TFIDF_MAX_FEATURES    = 50_000

# ──────────────────────────────────────────────────────────────────────────────
# UI colours
# ──────────────────────────────────────────────────────────────────────────────
COLOR_REAL  = "#22c55e"
COLOR_FAKE  = "#ef4444"
COLOR_ACCENT = "#6ee7f9"
COLOR_PURPLE = "#a78bfa"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e5e7eb", family="Trebuchet MS, sans-serif"),
    margin=dict(l=30, r=20, t=40, b=30),
)
