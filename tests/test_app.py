"""Clicks through the Streamlit app in demo mode."""
from conftest import ROOT
from streamlit.testing.v1 import AppTest


def test_app_runs_both_tools_in_demo_mode():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    at.button(key=None) if False else None
    buttons = {b.label: b for b in at.button}
    buttons["Use the sample datasheet"].click().run()
    {b.label: b for b in at.button}["Analyse datasheet"].click().run()
    assert not at.exception and not at.error
    assert any("Register map" in m.value for m in at.markdown)
    {b.label: b for b in at.button}["Use the sample notes"].click().run()
    {b.label: b for b in at.button}["Analyse notes"].click().run()
    assert not at.exception and not at.error
    assert any("Shift-meeting report" in m.value for m in at.markdown)
