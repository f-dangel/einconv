"""Run the tutorials."""

from os import path
from subprocess import run

HERE = path.abspath(__file__)
HEREDIR = path.dirname(HERE)
TUTORIAL_DIR = path.join(path.dirname(HEREDIR), "docs", "tutorials")


def test_basic_conv2d():
    """Execute basic example that compares 2d convolution."""
    filename = path.join(TUTORIAL_DIR, "basic_conv2d.py")
    run(["python", filename], check=True)
