"""Portable interactive reports for completed benchmark workflows."""

from .calibration import render_calibration_report
from .comparison import render_comparison_report

__all__ = ["render_calibration_report", "render_comparison_report"]
