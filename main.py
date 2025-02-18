#!/usr/bin/env python3
"""
This script captures screenshots from a specified monitor,
processes them using template matching and OCR (via PaddleOCR),
and then displays results in a PyQt6 GUI. It also contains logic
to place trading orders using Alpaca's API based on OCR output.

Modules used:
- sys, time: Standard libraries for system and time-related operations.
- multiprocessing: To run capture (producer) and processing (consumer) concurrently.
- numpy, cv2: For image processing.
- mss: For screen capturing.
- re, logging: For regex operations and logging.
- paddleocr: For performing OCR on captured images.
- asyncio, qasync: For asynchronous GUI updates and Alpaca order placement.
- PyQt6: For creating the GUI.
- alpaca.trading: For trading order placements via Alpaca.
- dotenv: For loading environment variables.
"""

import sys
import time
import multiprocessing as mp
import numpy as np
import cv2
import mss
import re
import logging
import asyncio

# Third-party libraries for OCR and asynchronous GUI support
from paddleocr import PaddleOCR
from qasync import asyncSlot

# PyQt6 modules for GUI components
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QSizePolicy, QComboBox, QSpacerItem,
    QVBoxLayout, QWidget, QPushButton, QHBoxLayout
)
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

# Alpaca trading client imports for order placements
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# dotenv for environment variable management
from dotenv import load_dotenv
import os

# -----------------------------------------------------------------------------
# Configuration and Environment Setup
# -----------------------------------------------------------------------------

# Load environment variables from .env file
load_dotenv()

# Configure logging settings
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("ppocr").setLevel(logging.ERROR)

# Alpaca API settings: Retrieve API keys and order URL from environment
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")        # Your Alpaca API key
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")    # Your Alpaca secret key
ALPACA_ORDERS_URL = os.getenv("ALPACA_ORDERS_URL")    # Alpaca orders endpoint URL

# Initialize the Alpaca trading client (using paper trading endpoint)
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

# -----------------------------------------------------------------------------
# Asynchronous Alpaca Order Placement Function
# -----------------------------------------------------------------------------

async def place_alpaca_order(ticker, order_qty, side):
    """
    Places a market order using Alpaca's API.
    - ticker: Stock symbol
    - order_qty: Quantity to buy or sell
    - side: 'buy' or 'sell'
    """
    try:
        # Attempt to retrieve current position
        try:
            position = await trading_client.get_position(ticker)
            current_qty = float(position.qty)
        except Exception:
            current_qty = 0

        # Check for sufficient shares when selling
        if side.lower() == "sell" and current_qty < order_qty:
            logging.error(f"Cannot sell {order_qty} shares of {ticker}; only {current_qty} available.")
            return

        # Determine order side based on string input
        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

        # Create a market order request with Good-Til-Canceled time in force
        order_request = MarketOrderRequest(
            symbol=ticker,
            qty=order_qty,
            side=order_side,
            time_in_force=TimeInForce.GTC
        )

        # Submit the order asynchronously
        order = await trading_client.submit_order(order_data=order_request)
        logging.info(f"Alpaca {side} order placed: {order}")
    except Exception as e:
        logging.error(f"Error placing Alpaca {side} order: {e}")

# -----------------------------------------------------------------------------
# OCR Initialization
# -----------------------------------------------------------------------------

def init_ocr():
    """
    Initializes a PaddleOCR instance configured for English without angle classification,
    and uses GPU processing if available.
    """
    return PaddleOCR(use_angle_cls=False, lang='en', use_gpu=True)

# -----------------------------------------------------------------------------
# Producer Function: Screen Capture
# -----------------------------------------------------------------------------

def producer(frame_queue, running, monitor_number=1):
    """
    Continuously captures screenshots from the specified monitor using mss.
    Captured frames (along with capture timing and timestamp) are put into a queue.
    """
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[monitor_number]
            logging.info(f"Producer started, monitoring: {monitor}")
            while running.value:
                try:
                    capture_start = time.perf_counter()
                    screenshot = sct.grab(monitor)
                    frame = np.array(screenshot)
                    # Convert from BGRA to BGR color space
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    capture_end = time.perf_counter()
                    capture_time = (capture_end - capture_start) * 1000  # in milliseconds

                    try:
                        frame_queue.put_nowait((frame, capture_time, time.time()))
                    except mp.queues.Full:
                        logging.warning("Producer: Frame queue is full, skipping frame")
                except Exception as e:
                    logging.error(f"Producer error during capture: {e}")
                    time.sleep(0.1)
            logging.info("Producer stopped")
    except Exception as e:
        logging.critical(f"Producer failed to initialize: {e}")

# -----------------------------------------------------------------------------
# Consumer Function: Image Processing & OCR
# -----------------------------------------------------------------------------

def consumer(frame_queue, result_queue, running):
    """
    Processes frames from the frame_queue by:
    - Performing template matching to locate a region of interest.
    - Extracting three Regions Of Interest (ROIs) based on preset offsets.
    - Stacking these ROIs and running OCR on the merged image.
    - Parsing OCR results into ticker, quantity, and position/flat values.
    - Pushing results and timing data to the result_queue.
    """
    try:
        ocr = init_ocr()
        template_found = False
        template_coordinates = None

        # Load the template image (make sure 'template.png' exists in the working directory)
        template = cv2.imread("template.png")
        if template is None:
            logging.error("Template image not found. Please provide 'template.png'.")
            return

        # Define offsets for ROI extraction: (x_offset, y_offset, width, height)
        ticker_symbol_offset = (-460, -75, 62, 18)
        qty_value_offset = (-120, -25, 62, 18)
        pos_flat_offset = (-50, 50, 62, 18)

        logging.info("Consumer started")

        while running.value:
            try:
                if not frame_queue.empty():
                    frame, capture_time, timestamp = frame_queue.get_nowait()
                    screen_np = frame

                    # Perform template matching if not already found
                    if not template_found:
                        result_template = cv2.matchTemplate(screen_np, template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, template_coordinates = cv2.minMaxLoc(result_template)
                        if max_val < 0.8:
                            logging.info("Template not found on screen.")
                            # Create a blank image and send a "not found" result to the GUI
                            blank_image = np.zeros((270, 310, 3), dtype=np.uint8)
                            result_queue.put((time.time(),
                                              ["Not Found", "Not Found", "Not Found"],
                                              capture_time, 0, 0, blank_image))
                            continue
                        else:
                            template_found = True
                            logging.info(f"Template found at: {template_coordinates}")

                    x, y = template_coordinates

                    # Helper function to crop and resize an ROI based on an offset tuple
                    def extract_roi(offset):
                        x1, y1, w, h = offset
                        roi = screen_np[y + y1: y + y1 + h, x + x1: x + x1 + w]
                        roi = cv2.resize(roi, (310, 90))
                        return roi

                    # Extract ROIs for ticker, quantity, and position/flat values
                    ticker_roi = extract_roi(ticker_symbol_offset)
                    qty_roi = extract_roi(qty_value_offset)
                    pos_flat_roi = extract_roi(pos_flat_offset)

                    # Merge the three ROIs vertically (total height = 270)
                    merged_roi = np.vstack([ticker_roi, qty_roi, pos_flat_roi])

                    # Run OCR on the merged image and record processing time
                    ocr_start = time.perf_counter()
                    try:
                        ocr_results = ocr.ocr(merged_roi, cls=True)
                        if not ocr_results:
                            raise ValueError("Empty OCR result")
                    except Exception as e:
                        logging.error(f"OCR processing failed: {e}")
                        ocr_results = []
                    ocr_end = time.perf_counter()
                    processing_time = (ocr_end - ocr_start) * 1000

                    # Parse OCR output into three rows based on vertical position
                    rows = {0: [], 1: [], 2: []}
                    for line in ocr_results:
                        for detection in line:
                            box, (text, confidence) = detection
                            top_y = min(pt[1] for pt in box)
                            if top_y < 90:
                                row_index = 0
                            elif top_y < 180:
                                row_index = 1
                            else:
                                row_index = 2
                            rows[row_index].append(text)

                    # Combine OCR results for each row
                    ticker_text = ' '.join(rows[0]) if rows[0] else "No text"
                    qty_text = re.sub(r'\D', '', ' '.join(rows[1]) if rows[1] else "0")
                    pos_flat_text = re.sub(r'\D', '', ' '.join(rows[2]) if rows[2] else "0")
                    ocr_result = [ticker_text, qty_text, pos_flat_text]

                    total_latency = (time.perf_counter() - timestamp) * 1000

                    # Put results (including OCR text, timing data, and image) into result_queue
                    result_queue.put((time.time(), ocr_result, capture_time, processing_time, total_latency, merged_roi))
                else:
                    # Sleep briefly to avoid busy-waiting if queue is empty
                    time.sleep(0.001)
            except mp.queues.Empty:
                continue
            except Exception as e:
                logging.error(f"Consumer error during processing: {e}")
                time.sleep(0.1)
        logging.info("Consumer stopped")
    except Exception as e:
        logging.critical(f"Consumer failed to initialize: {e}")

# -----------------------------------------------------------------------------
# UpdateThread: QThread for GUI Updates
# -----------------------------------------------------------------------------

class UpdateThread(QThread):
    """
    A QThread subclass that continuously checks the result_queue for new data
    and emits a signal to update the GUI.
    """
    result_signal = pyqtSignal(tuple)  # Signal carrying the OCR result data

    def __init__(self, result_queue):
        super().__init__()
        self.result_queue = result_queue
        self.running = True

    def run(self):
        logging.info("UpdateThread started")
        while self.running:
            try:
                if not self.result_queue.empty():
                    data = self.result_queue.get_nowait()
                    self.result_signal.emit(data)
                else:
                    time.sleep(0.001)
            except mp.queues.Empty:
                continue
            except Exception as e:
                logging.error(f"UpdateThread error: {e}")
                time.sleep(0.1)
        logging.info("UpdateThread stopped")

    def stop(self):
        """Stops the update thread."""
        self.running = False
        self.quit()
        self.wait()

# -----------------------------------------------------------------------------
# MainWindow: PyQt6 GUI Class
# -----------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """
    Main window of the application which displays OCR results, timing data,
    and the processed image. It also manages the start/stop of background processes.
    """
    def __init__(self, frame_queue, result_queue, running):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.running = running  # Shared multiprocessing flag
        self.prod_process = None
        self.consumer_process = None
        self.update_thread = None

        # Setup window properties
        self.setWindowTitle("OCR Results")
        self.resize(400, 500)

        # Create labels to display OCR text results
        self.ticker_label = QLabel("Ticker: ")
        self.qty_label = QLabel("Quantity: ")
        self.pos_label = QLabel("Position/Flat: ")

        # Label to display the merged ROI image (for visual feedback)
        self.image_label = QLabel()
        self.image_label.setFixedSize(310, 270)

        # Labels to display timing information
        self.capture_time_label = QLabel("Capture Time: ")
        self.ocr_time_label = QLabel("OCR Time: ")
        self.total_latency_label = QLabel("Total Latency: ")

        # Divider layout with a combo box to select a divider value (1 to 100)
        divider_layout = QHBoxLayout()
        self.label_divider = QLabel("Divider Value")
        divider_layout.addWidget(self.label_divider)
        self.horizontalSpacer_2 = QSpacerItem(39, 20, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        divider_layout.addItem(self.horizontalSpacer_2)
        self.divider_value = QComboBox()
        for i in range(1, 101):
            self.divider_value.addItem(str(i))
        divider_layout.addWidget(self.divider_value)
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        divider_layout.addItem(self.horizontalSpacer)

        # Create buttons for controlling the application
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.exit_button = QPushButton("Exit")
        self.start_button.clicked.connect(self.start_processes)
        self.stop_button.clicked.connect(self.stop_processes)
        self.exit_button.clicked.connect(self.close)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        # Uncomment the next line to add a Stop button to the layout if needed
        # button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.exit_button)

        # Overall layout: combine all widgets and layouts
        layout = QVBoxLayout()
        layout.addWidget(self.ticker_label)
        layout.addWidget(self.qty_label)
        layout.addWidget(self.pos_label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.capture_time_label)
        layout.addWidget(self.ocr_time_label)
        layout.addWidget(self.total_latency_label)
        layout.addLayout(divider_layout)
        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Initially disable the Stop button
        self.stop_button.setEnabled(False)

    @asyncSlot(tuple)
    async def handle_update(self, data):
        """Slot to handle updates from the UpdateThread."""
        await self.update_gui(data)

    async def update_gui(self, data):
        """
        Updates the GUI with new OCR data, timing information, and image display.
        Also determines if a trading order should be placed based on OCR results.
        """
        try:
            if len(data) == 6:
                timestamp, result, capture_time, ocr_time, total_latency, merged_roi = data
                ticker_text, qty_text, pos_text = result

                # Update text labels with OCR results
                self.ticker_label.setText(f"Ticker: {ticker_text}")
                self.qty_label.setText(f"Quantity: {qty_text}")
                self.pos_label.setText(f"Position/Flat: {pos_text}")

                # Determine action based on OCR results using predefined cases
                cases = {
                    ("not_4500", "0"): lambda: print("Nothing"),
                    ("4500", "4500"): lambda: asyncio.create_task(place_alpaca_order(ticker_text, 4500 / int(self.divider_value.currentText()), "buy")),
                    ("4500", "0"): lambda: asyncio.create_task(place_alpaca_order(ticker_text, 4500 / int(self.divider_value.currentText()), "sell")),
                    ("2700", "0"): lambda: print("Nothing"),
                    ("2700", "2700"): lambda: print("Nothing"),
                    ("900", "1800"): lambda: asyncio.create_task(place_alpaca_order(ticker_text, 900 / int(self.divider_value.currentText()), "sell")),
                    ("2700", "5400"): lambda: asyncio.create_task(place_alpaca_order(ticker_text, 900 / int(self.divider_value.currentText()), "buy")),
                    ("1800", "0"): lambda: asyncio.create_task(place_alpaca_order(ticker_text, 1800 / int(self.divider_value.currentText()), "sell")),
                }
                # Determine key based on OCR quantity text and position/flat text
                if qty_text != "4500":
                    action_key = ("not_4500", pos_text)
                else:
                    action_key = (qty_text, pos_text)
                action = cases.get(action_key, lambda: print("No match"))
                action()

                # Convert merged ROI (BGR image) to QImage for display in QLabel
                height, width, channel = merged_roi.shape
                bytesPerLine = 3 * width
                qimg = QImage(merged_roi.data, width, height, bytesPerLine, QImage.Format.Format_BGR888)
                pixmap = QPixmap.fromImage(qimg)
                self.image_label.setPixmap(pixmap)

                # Update timing labels with formatted times
                self.capture_time_label.setText(f"Capture Time: {capture_time:.2f} ms")
                self.ocr_time_label.setText(f"OCR Time: {ocr_time:.2f} ms")
                self.total_latency_label.setText(f"Total Latency: {total_latency:.2f} ms")
            else:
                logging.warning(f"update_gui: Unexpected data format: {data}")
        except Exception as e:
            logging.error(f"Error in update_gui: {e}")

    def start_processes(self):
        """
        Starts the producer (screen capture), consumer (processing), and update thread (GUI updates).
        Adjusts button states accordingly.
        """
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.running.value = True  # Signal to start processes
        logging.info("Starting processes...")

        # Create new multiprocessing processes for producer and consumer
        self.prod_process = mp.Process(target=producer, args=(self.frame_queue, self.running))
        self.consumer_process = mp.Process(target=consumer, args=(self.frame_queue, self.result_queue, self.running))
        self.update_thread = UpdateThread(self.result_queue)

        # Connect update signal from the thread to the GUI update slot
        self.update_thread.result_signal.connect(self.handle_update)

        # Start processes and thread
        self.prod_process.start()
        self.consumer_process.start()
        self.update_thread.start()
        logging.info("Processes started") 

    def stop_processes(self):
        """
        Stops the producer, consumer, and update thread gracefully.
        Resets button states and clears remaining data in queues.
        """
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        logging.info("Stopping processes...")
        self.running.value = False  # Signal processes to stop

        # Stop and terminate update thread if running
        if self.update_thread:
            self.update_thread.stop()
            self.update_thread.terminate()
            self.update_thread = None

        # Gracefully join producer process
        if self.prod_process:
            self.prod_process.join(timeout=2)
            if self.prod_process.is_alive():
                self.prod_process.terminate()
            self.prod_process = None

        # Gracefully join consumer process
        if self.consumer_process:
            self.consumer_process.join(timeout=2)
            if self.consumer_process.is_alive():
                self.consumer_process.terminate()
            self.consumer_process = None

        # Clear any residual data in the queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except mp.queues.Empty:
                break
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except mp.queues.Empty:
                break

        logging.info("Processes stopped and queues cleared")

    def closeEvent(self, event):
        """Handle the closing event by stopping background processes before exit."""
        self.stop_processes()
        super().closeEvent(event)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    try:
        # Use 'spawn' start method for compatibility (especially on Windows)
        mp.set_start_method("spawn")

        # Create multiprocessing queues for frames and OCR results
        frame_queue = mp.Queue(maxsize=5)  # Limited size for near real-time performance
        result_queue = mp.Queue()

        # Shared flag for controlling running state
        running = mp.Value('b', False)  # Boolean flag, initially False

        # Initialize the PyQt6 application and integrate with qasync event loop
        app = QApplication(sys.argv)
        import qasync
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)

        with loop:
            main_window = MainWindow(frame_queue, result_queue, running)
            main_window.show()
            loop.run_forever()
    finally:
        loop.close()
