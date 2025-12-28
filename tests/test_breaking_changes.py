"""Unit tests for breaking changes database."""

import pytest
from src.data.breaking_changes import (
    BreakingChange,
    BreakingChangesDatabase,
    Severity,
    score_compatibility,
    get_breaking_changes,
)


class TestBreakingChangesDatabase:
    """Test suite for BreakingChangesDatabase."""

    def setup_method(self):
        """Set up test fixtures."""
        self.db = BreakingChangesDatabase()

    def test_database_initialization(self):
        """Test database initializes with changes."""
        changes = self.db.get_all_changes()

        assert len(changes) > 0
        assert all(isinstance(c, BreakingChange) for c in changes)

    def test_get_changes_by_library(self):
        """Test filtering changes by library."""
        pytorch_changes = self.db.get_changes_by_library("pytorch")

        assert len(pytorch_changes) > 0
        assert all(c.affected_library == "pytorch" for c in pytorch_changes)

    def test_get_changes_by_cuda_transition(self):
        """Test filtering by CUDA version transition."""
        changes = self.db.get_changes_by_cuda_transition("12.4", "13.0")

        assert len(changes) > 0

    def test_score_compatibility_no_issues(self):
        """Test compatibility scoring with no issues."""
        libraries = [{"name": "pytorch", "version": "2.1.0", "cuda_version": "12.4"}]

        score = self.db.score_compatibility(libraries, "12.4")

        # Should have high score with no breaking changes
        assert score["compatibility_score"] >= 90
        assert score["critical_issues"] == 0

    def test_score_compatibility_with_critical(self):
        """Test compatibility scoring with critical issues."""
        libraries = [{"name": "pytorch", "version": "2.1.0", "cuda_version": "12.4"}]

        score = self.db.score_compatibility(libraries, "13.0")

        # Should detect PyTorch CUDA 13.0 incompatibility
        assert score["critical_issues"] > 0
        assert score["compatibility_score"] < 100

    def test_score_compatibility_compute_capability(self):
        """Test scoring with compute capability checks."""
        libraries = [
            {"name": "tensorflow", "version": "2.15.0", "cuda_version": "12.1"}
        ]

        score = self.db.score_compatibility(libraries, "12.1", compute_capability="9.0")

        # Should detect TensorFlow SM_90 issue
        assert score["critical_issues"] > 0

    def test_recommendation_critical(self):
        """Test recommendation with critical issues."""
        recommendation = self.db._get_recommendation(50, critical_count=1)

        assert "CRITICAL" in recommendation

    def test_recommendation_good(self):
        """Test recommendation with high score."""
        recommendation = self.db._get_recommendation(95, critical_count=0)

        assert "GOOD" in recommendation

    def test_export_to_json(self, tmp_path):
        """Test exporting database to JSON."""
        output_file = tmp_path / "test_changes.json"
        self.db.export_to_json(str(output_file))

        assert output_file.exists()

        # Verify can load it back
        self.db.load_from_json(str(output_file))
        assert len(self.db.get_all_changes()) > 0


def test_score_compatibility_function():
    """Test convenience function for scoring."""
    libraries = [{"name": "pytorch", "version": "2.1.0", "cuda_version": "12.4"}]

    score = score_compatibility(libraries, "12.4")

    assert "compatibility_score" in score
    assert "recommendation" in score


def test_get_breaking_changes_all():
    """Test getting all breaking changes."""
    changes = get_breaking_changes()

    assert len(changes) > 0
    assert all(isinstance(c, dict) for c in changes)


def test_get_breaking_changes_filtered():
    """Test getting filtered breaking changes."""
    changes = get_breaking_changes(library="cudf")

    assert len(changes) > 0
    assert all(c["affected_library"] == "cudf" for c in changes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

