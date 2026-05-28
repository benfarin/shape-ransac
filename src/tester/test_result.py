from dataclasses import dataclass

from src.tester.match_result import MatchResult


@dataclass
class TestResult:
    matches: list[MatchResult]
    mean_error: float
    median_error: float
