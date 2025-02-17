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

project_dir = Path(__file__).parents[2]
result_dir = Path(__file__).parents[2] / "result"
chart_dir = Path(__file__).parents[2] / "chart"
score_dir = Path(__file__).parents[2] / "score"
data_dir = Path(__file__).parents[2] / "data"
