
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_app_displays_title():
    """The app should display its main title."""

    at = AppTest.from_file(APP_PATH).run()

    assert not at.exception
    assert at.title[0].value == "Financial Complaint Classifier"


def test_app_rejects_empty_complaint():
    """An empty complaint should display a warning."""

    at = AppTest.from_file(APP_PATH).run()

    at.button[0].click().run()

    assert not at.exception
    assert len(at.warning) == 1
    assert "Please enter a complaint" in at.warning[0].value


def test_app_contains_complaint_input():
    """The app should provide an input for complaint text."""

    at = AppTest.from_file(APP_PATH).run()

    assert not at.exception
    assert len(at.text_area) == 1
    assert at.text_area[0].label == "Enter your financial complaint"
