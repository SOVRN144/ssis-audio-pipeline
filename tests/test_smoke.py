"""Smoke test to verify testing infrastructure is working."""


def test_smoke():
    """Placeholder test that always passes to ensure CI is green in Step 0."""
    assert True


def test_python_version():
    """Verify Python version meets requirements."""
    import sys

    assert sys.version_info >= (3, 11), "Python 3.11 or higher required"
