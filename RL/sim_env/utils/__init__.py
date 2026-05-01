"""
仿真环境工具模块

该模块包含仿真环境的辅助工具和实用功能：
- reliability_evaluation: 基于统计分析的可靠性评价功能
"""

from .reliability_evaluation import SlidingWindowTamperDetector

__all__ = ['SlidingWindowTamperDetector']