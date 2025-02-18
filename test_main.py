import sys
import pytest
from PyQt6.QtWidgets import QApplication
from main import OCRApp

# FILE: test_main.py


@pytest.fixture(scope="module")
def app():
    """Fixture to create the QApplication instance."""
    app = QApplication(sys.argv)
    yield app
    app.quit()

def test_ocr_app_initialization(app):
    """Test the initialization of OCRApp."""
    window = OCRApp()
    
    # Check window title
    assert window.windowTitle() == "OCR App"
    
    # Check window geometry
    assert window.geometry().x() == 100
    assert window.geometry().y() == 100
    assert window.geometry().width() == 600
    assert window.geometry().height() == 500
    
    # Check if OCRWorker is initialized
    assert window.ocr_worker is not None
    
    # Check if update_signal is connected to update_ui
    assert window.ocr_worker.update_signal.receivers(window.update_ui) > 0
    
    # Check if buttons are initialized and connected
    assert window.btn_start.text() == "Start"
    assert window.btn_stop.text() == "Stop"
    assert window.btn_exit.text() == "Exit"
    
    # Check if labels are initialized
    assert window.label_ticker.text() == "Ticker Symbol"
    assert window.label_qty.text() == "Qty Value"
    assert window.label_pos.text() == "Pos/Flat Value"
    assert window.label_divider.text() == "Divider Value"
    assert window.time_label.text() == "Processing Time: 0s"
    
    # Check if combo box is initialized with correct values
    assert window.divider_value.count() == 20
    for i in range(1, 21):
        assert window.divider_value.itemText(i-1) == str(i)
    
    # Check if buttons are enabled/disabled correctly
    assert window.btn_start.isEnabled() is True
    assert window.btn_stop.isEnabled() is False
