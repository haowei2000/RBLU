"""
This module provides directory path constants for the project.

Attributes:
    project_dir (Path): The root directory of the project.
    result_dir (Path): The directory for storing result files.
    chart_dir (Path): The directory for storing chart files.
    score_dir (Path): The directory for storing score files.
    data_dir (Path): The directory for storing data files.
"""

from pathlib import Path

CONFIG_PATH = Path(__file__).parents[1] / "config.yml"
PROJECT_DIR = Path(__file__).parents[2]
RESULT_DIR = Path(__file__).parents[2] / "result"
CHART_DIR = Path(__file__).parents[2] / "chart"
SCORE_DIR = Path(__file__).parents[2] / "score"
DATA_DIR = Path(__file__).parents[2] / "data"
