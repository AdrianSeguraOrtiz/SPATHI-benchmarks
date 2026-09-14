"""Portable interactive reports for completed benchmark workflows."""

from .calibration import render_calibration_report
from .comparison import render_comparison_report
from .dimensional_transfer import render_dimensional_transfer_report
from .performance import render_performance_report
from .scalability import render_scalability_report

__all__ = [
    "render_calibration_report",
    "render_comparison_report",
    "render_dimensional_transfer_report",
    "render_performance_report",
    "render_scalability_report",
]
