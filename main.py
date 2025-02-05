import sys
import cv2
import numpy as np
import pyautogui
import time
import logging
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QComboBox,
    QSpacerItem,
    QSizePolicy,
)
from paddleocr import PaddleOCR

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("ppocr").setLevel(logging.ERROR)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)

# Load the template
template = cv2.imread("template.png")
if template is None:
    logging.error("Template image not found or could not be loaded.")
    sys.exit(1)

template_h, template_w, _ = template.shape

# Define offsets for different fields
ticker_symbol_offset = (-480, -80, 62, 18)
qty_value_offset = (-70, -20, 62, 18)
pos_flat_offset = (50, 57, 62, 18)

class OCRWorker(QThread):
    update_signal = pyqtSignal(str, str, str, float, np.ndarray, np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = False
        self.template_found = False

    def run(self):
        while self.running:
            try:
                start_time = time.time()
                screen = pyautogui.screenshot()
                screen_np = np.array(screen)

                # Template matching
                if self.template_found is False:
                    result = cv2.matchTemplate(screen_np, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)

                    if max_val < 0.8:
                        logging.info("Template not found on screen.")
                        self.update_signal.emit(
                            "Not Found",
                            "Not Found",
                            "Not Found",
                            0,
                            np.zeros((90, 310, 3), dtype=np.uint8),
                            np.zeros((90, 310, 3), dtype=np.uint8),
                            np.zeros((90, 310, 3), dtype=np.uint8),
                        )
                        time.sleep(1)
                        continue
                    self.template_found = True

                x, y = max_loc

                # Function to extract ROI using offset
                def extract_roi(offset):
                    x1, y1, w, h = offset
                    roi = screen_np[y + y1 : y + y1 + h, x + x1 : x + x1 + w]
                    roi = cv2.resize(roi, (310, 90))
                    return roi

                ticker_roi = extract_roi(ticker_symbol_offset)
                qty_roi = extract_roi(qty_value_offset)
                pos_flat_roi = extract_roi(pos_flat_offset)

                rows = {0: [], 1: [], 2: []}

                merged_roi = np.vstack([ticker_roi, qty_roi, pos_flat_roi])

                # Run OCR on the merged ROI
                try:
                    ocr_results = ocr.ocr(merged_roi, cls=True)
                    if not ocr_results:
                        raise ValueError("Empty OCR result")
                except Exception as e:
                    logging.error(f"OCR processing failed: {e}")
                    ocr_results = []

                # PaddleOCR returns a list of lists where each sublist corresponds to one detected text line.
                # Each detection is of the form [box, (text, confidence)].
                for line in ocr_results:
                    for detection in line:
                        box, (text, confidence) = detection
                        # Determine the top y coordinate of the detected box.
                        top_y = min(pt[1] for pt in box)
                        # Determine which row the text belongs to.
                        if top_y < 90:
                            row_index = 0
                        elif top_y < 180:
                            row_index = 1
                        else:
                            row_index = 2
                        rows[row_index].append(text)

                # Combine texts in each row (if there are multiple detections per ROI)
                ticker_text   = ' '.join(rows[0]) if rows[0] else "No text"
                qty_text      = ' '.join(rows[1]) if rows[1] else "No text"
                pos_flat_text = ' '.join(rows[2]) if rows[2] else "No text"

                processing_time = time.time() - start_time
                self.update_signal.emit(ticker_text, qty_text, pos_flat_text, processing_time, ticker_roi, qty_roi, pos_flat_roi)
            except Exception as e:
                logging.error(f"Error in OCRWorker thread: {e}")
            time.sleep(1)

    def start_ocr(self):
        self.running = True
        self.start()

    def stop_ocr(self):
        self.running = False
        self.template_found = False
        self.quit()

class OCRApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR App")
        self.setGeometry(100, 100, 600, 500)

        self.ocr_worker = OCRWorker()
        self.ocr_worker.update_signal.connect(self.update_ui)

        layout = QVBoxLayout()

        font = QFont()
        font.setPointSize(11)
        font.setBold(True)

        # Ticker Symbol layout
        ticker_layout = QHBoxLayout()
        self.label_ticker = QLabel("Ticker Symbol")
        self.label_ticker.setFont(font)
        ticker_layout.addWidget(self.label_ticker)

        self.label_ticker_value = QLabel("SCAP")
        self.label_ticker_value.setFont(font)
        ticker_layout.addWidget(self.label_ticker_value)

        self.label_ticker_roi = QLabel()
        self.label_ticker_roi.setFixedSize(310, 90)
        ticker_layout.addWidget(self.label_ticker_roi)

        layout.addLayout(ticker_layout)

        # Quantity layout
        qty_layout = QHBoxLayout()
        self.label_qty = QLabel("Qty Value")
        self.label_qty.setFont(font)
        qty_layout.addWidget(self.label_qty)

        self.label_qty_value = QLabel("+4500")
        self.label_qty_value.setFont(font)
        qty_layout.addWidget(self.label_qty_value)

        self.label_qty_roi = QLabel()
        self.label_qty_roi.setFixedSize(310, 90)
        qty_layout.addWidget(self.label_qty_roi)

        layout.addLayout(qty_layout)

        # Position/Flat layout
        pos_layout = QHBoxLayout()
        self.label_pos = QLabel("Pos/Flat Value")
        self.label_pos.setFont(font)
        pos_layout.addWidget(self.label_pos)

        self.label_pos_value = QLabel("0")
        self.label_pos_value.setFont(font)
        pos_layout.addWidget(self.label_pos_value)

        self.label_pos_roi = QLabel()
        self.label_pos_roi.setFixedSize(310, 90)
        pos_layout.addWidget(self.label_pos_roi)

        layout.addLayout(pos_layout)

        # Divider selection layout
        divider_layout = QHBoxLayout()
        self.label_divider = QLabel("Divider Value")
        self.label_divider.setFont(font)
        divider_layout.addWidget(self.label_divider)

        self.horizontalSpacer_2 = QSpacerItem(39, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        divider_layout.addItem(self.horizontalSpacer_2)

        self.divider_value = QComboBox()
        self.divider_value.setFont(font)
        for i in range(1, 21):
            self.divider_value.addItem(str(i))
        divider_layout.addWidget(self.divider_value)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        divider_layout.addItem(self.horizontalSpacer)

        layout.addLayout(divider_layout)

        # Processing time label
        self.time_label = QLabel("Processing Time: 0s")
        layout.addWidget(self.time_label)

        # Buttons layout
        button_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.start_ocr)
        button_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_ocr)
        button_layout.addWidget(self.btn_stop)

        self.btn_exit = QPushButton("Exit")
        self.btn_exit.clicked.connect(self.close)
        button_layout.addWidget(self.btn_exit)

        self.btn_stop.setEnabled(False)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def start_ocr(self):
        self.ocr_worker.start_ocr()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_ocr(self):
        self.ocr_worker.stop_ocr()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def update_ui(self, ticker_text, qty_text, pos_flat_text, processing_time, ticker_roi, qty_roi, pos_flat_roi):
        self.label_ticker_value.setText(ticker_text)
        self.label_qty_value.setText(qty_text)
        self.label_pos_value.setText(pos_flat_text)
        self.time_label.setText(f"Processing Time: {processing_time:.3f}s")

        # Helper function to update a QLabel with an OpenCV image
        def set_label_image(label, roi):
            try:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                height, width, channel = roi_rgb.shape
                bytes_per_line = 3 * width
                q_img = QImage(roi_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                label.setPixmap(pixmap)
            except Exception as e:
                logging.error(f"Error updating image label: {e}")

        set_label_image(self.label_ticker_roi, ticker_roi)
        set_label_image(self.label_qty_roi, qty_roi)
        set_label_image(self.label_pos_roi, pos_flat_roi)

if __name__ == "__main__":
    # Enable High DPI scaling to avoid DPI related warnings on Windows
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_, True)

    app = QApplication(sys.argv)
    window = OCRApp()
    window.show()
    sys.exit(app.exec())
