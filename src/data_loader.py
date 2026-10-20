"""
data_loader.py — Dataset loading, session management, and sample data generator.
"""
from __future__ import annotations
import io
from pathlib import Path
from typing import Any
import pandas as pd
import streamlit as st

from src.config import DATA_DIR


def get_session_dataset() -> pd.DataFrame | None:
    """Retrieve current working dataset from Streamlit session state."""
    return st.session_state.get("active_df", None)


def set_session_dataset(df: pd.DataFrame, name: str = "dataset.csv", metadata: dict | None = None):
    """Save active dataset and metadata into Streamlit session state."""
    st.session_state["active_df"] = df
    st.session_state["active_df_name"] = name
    st.session_state["active_df_meta"] = metadata or {}


def clear_session_dataset():
    """Remove current working dataset from session state."""
    st.session_state.pop("active_df", None)
    st.session_state.pop("active_df_name", None)
    st.session_state.pop("active_df_meta", None)


def load_csv(file_or_path: Any, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Load a CSV file with automatic fallback on common encoding issues.
    """
    try:
        if isinstance(file_or_path, (str, Path)):
            return pd.read_csv(file_or_path, encoding=encoding)
        else:
            return pd.read_csv(file_or_path, encoding=encoding)
    except UnicodeDecodeError:
        # Fallback to latin-1 / cp1252 if utf-8 fails
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
        return pd.read_csv(file_or_path, encoding="latin-1")


def create_sample_research_dataset() -> pd.DataFrame:
    """
    Generates a realistic synthetic research dataset for demonstration, testing,
    and immediate platform exploration without requiring external downloads.
    """
    records = [
        # Real news style
        ("The Federal Reserve held benchmark interest rates steady following the conclusion of its two-day policy meeting, citing balanced inflation indicators and stable employment data.", "REAL"),
        ("Global climate diplomats convened in Geneva to draft standardized monitoring protocols for national carbon reduction commitments ahead of the UN summit.", "REAL"),
        ("NASA scientists announced new spectral analysis data gathered by the James Webb Space Telescope confirming atmospheric water vapor on an exoplanet 120 light years away.", "REAL"),
        ("The World Health Organization published updated guidelines on antimicrobial resistance surveillance, emphasizing regional laboratory cooperation.", "REAL"),
        ("Ministry of Transportation officials released the annual infrastructure safety report, showing significant decreases in railway transit incidents over the past decade.", "REAL"),
        ("European central banks agreed to coordinate stress tests evaluating how commercial lenders would absorb hypothetical liquidity shocks.", "REAL"),
        ("The Supreme Court heard oral arguments on appellate procedural standards governing environmental impact assessments under federal statute.", "REAL"),
        ("Agricultural researchers reported that drought-resistant grain varieties produced 18 percent higher yields during field trials conducted across semi-arid testing zones.", "REAL"),
        ("International maritime authorities reported a peaceful resolution to territorial navigation discussions following multilateral diplomatic talks.", "REAL"),
        ("The Department of Labor released employment statistics showing modest employment growth led by healthcare, construction, and educational sectors.", "REAL"),
        ("Archaeological teams working in the Mediterranean uncovered a well-preserved Roman civic basilica dating back to the late second century.", "REAL"),
        ("Biochemists at leading research universities synthesized a novel enzyme capable of decomposing industrial polyurethane polymers within days.", "REAL"),
        ("The Bureau of Meteorology confirmed that Pacific oceanic indicators indicate neutral ENSO conditions persisting through the upcoming autumn season.", "REAL"),
        ("City municipal planners approved a twenty-year urban greenway expansion project aiming to link municipal reservoirs with suburban park systems.", "REAL"),
        ("Telecommunications regulators issued new technical guidelines for next-generation spectrum allocations to support emergency service dispatch networks.", "REAL"),
        ("National health agencies released epidemiological data indicating seasonal influenza vaccination coverage had reached historical target thresholds.", "REAL"),
        ("Civil aviation investigators completed the structural inspection report regarding runway maintenance schedules at major regional international hubs.", "REAL"),
        ("Economists at the international trade commission released projections outlining export trends in clean energy machinery and automotive components.", "REAL"),
        ("Education board representatives adopted standardized digital literacy curricula for secondary school students across participating public districts.", "REAL"),
        ("Geological survey monitors recorded a minor magnitude 3.8 tremor along the offshore tectonic fault with no damage or tsunami danger reported.", "REAL"),

        # Fake news style
        ("SHOCKING EXPOSURE! Secret global cabal caught putting mind control nanobots in regular tap water! SHARE BEFORE DELETED!", "FAKE"),
        ("PROOF LEAKED: Hollywood elites use hidden underground tunnels to run secret dark-energy clone factories beneath theme parks!", "FAKE"),
        ("DOCTORS STUNNED: Miracle backyard weed destroys all disease in 24 hours but Big Pharma is banning it from the internet!", "FAKE"),
        ("ALERT: Government secretly signs treaty transferring complete national sovereignty to extraterrestrial council next Tuesday!", "FAKE"),
        ("BOMBSHELL: Famous celebrity reveals hidden billionaire ritual where politicians swap souls using ancient quantum mirrors!", "FAKE"),
        ("THEY DON'T WANT YOU TO KNOW: Drinking boiled banana peel water completely dissolves debt and clears your criminal record!", "FAKE"),
        ("LEAKED MILITARY DOCUMENTS show weather control storm towers are deliberately targeting harmless organic blueberry farms!", "FAKE"),
        ("INSIDER WHISTLEBLOWER reveals microwave ovens are actually surveillance devices recording your living room conversations for Martian spies!", "FAKE"),
        ("UNBELIEVABLE: Ancient pyramid found under Antarctica reveals mankind actually originated from subterranean magma people!", "FAKE"),
        ("BREAKING: Leading scientists admit the moon is an artificial holographic projector built by ancient time travelers in 1842!", "FAKE"),
        ("EXPOSED: Secret hidden camera footage proves election ballots were replaced by invisible edible rice paper duplicates!", "FAKE"),
        ("YOU WON'T BELIEVE THIS: Eating raw dandelion leaves reverses human aging by 40 years overnight according to banned guru!", "FAKE"),
        ("CONFIDENTIAL REPORT: Wireless 5G signals are transforming regular squirrels into aggressive remote-controlled spy agents!", "FAKE"),
        ("PANIC IN WASHINGTON: Mysterious glowing orb descends over capital building and demands all gold bars be surrendered immediately!", "FAKE"),
        ("MUST WATCH: Secret footage reveals ocean tides are actually controlled by giant underground water pumps installed in 1912!", "FAKE"),
        ("MAINSTREAM MEDIA SILENT: New viral photo shows world leaders having dinner with an eight-foot reptilian ambassador!", "FAKE"),
        ("HEALTH SCAM: Drinking water with lemon is an ancient hypnotic trigger designed to make citizens obey television advertisements!", "FAKE"),
        ("EXPLOSIVE DISCOVERY: Archaeologists uncover 5,000-year-old smartphone inside undisturbed Egyptian tomb battery still at 98%!", "FAKE"),
        ("WARNING ISSUED: New smart meters are emitting ultrasonic frequencies that erase your childhood memories while you sleep!", "FAKE"),
        ("HIDDEN TRUTH: Ancient scrolls prove gravity was only invented in 1687 by Isaac Newton to tax heavy household furniture!", "FAKE"),
    ]
    return pd.DataFrame(records, columns=["text", "label"])


def save_dataset_to_disk(df: pd.DataFrame, filename: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / filename
    df.to_csv(target, index=False)
    return target
