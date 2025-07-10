"""
This module provides directory path constants for the project.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).parents[2]
PROJECT_DIR = Path(__file__).parents[3]
RBLU_DIR = Path(__file__).parents[1]
CONFIG_PATH = RBLU_DIR / "config.yml"
RESULT_DIR = PACKAGE_DIR / "result"
CHART_DIR = PACKAGE_DIR / "chart"
SCORE_DIR = PACKAGE_DIR / "score"
DATA_DIR = PACKAGE_DIR / "data"
LOG_DIR = PROJECT_DIR / "log"
