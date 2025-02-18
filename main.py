import sys
import time
import multiprocessing as mp
import numpy as np
import cv2
import mss
import re
import logging
from paddleocr import PaddleOCR
import asyncio
import logging
from qasync import asyncSlot

from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QSizePolicy, QComboBox, QSpacerItem, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt6.QtCore import QThread, pyqtSignal

from PyQt6.QtGui import QImage, QPixmap

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from dotenv import load_dotenv
import os

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("ppocr").setLevel(logging.ERROR)

#Alpaca API setting:
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")        # Replace with your Alpaca API key
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")     # Replace with your Alpaca secret key
ALPACA_ORDERS_URL = os.getenv("ALPACA_ORDERS_URL") 
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True) # Using the paper trading endpoint

async def place_alpaca_order(ticker, order_qty, side):
    try:
        try:
            position = await trading_client.get_position(ticker)
            current_qty = float(position.qty)
        except Exception:
            current_qty = 0

        # If selling, ensure you have enough shares
        if side.lower() == "sell" and current_qty < order_qty:
            logging.error(f"Cannot sell {order_qty} shares of {ticker}; only {current_qty} available.")
            return
        # This converts the string 'buy' or 'sell' to the corresponding enum value.
        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

        # Create the market order request.
        # Here, MarketOrderRequest encapsulates all the details needed by Alpaca to process the order.
        order_request = MarketOrderRequest(
            symbol=ticker,
            qty=order_qty,
            side=order_side,
            time_in_force=TimeInForce.GTC  # Good-Til-Canceled order
        )

        # Submit the order asynchronously.
        # The async submit_order call sends the request to Alpaca's paper trading endpoint.
        order = await trading_client.submit_order(order_data=order_request)
        logging.info(f"Alpaca {side} order placed: {order}")
    except Exception as e:
        logging.error(f"Error placing Alpaca {side} order: {e}")

def init_ocr():
    """Initialize a single PaddleOCR instance for GPU processing."""
    return PaddleOCR(use_angle_cls=False, lang='en', use_gpu=True)

def producer(frame_queue, running, monitor_number=1):
    """Continuously captures screenshots as fast as possible using mss."""
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[monitor_number]
            logging.info(f"Producer started, monitoring: {monitor}")
            while running.value:
                try:
                    capture_start = time.perf_counter()
                    screenshot = sct.grab(monitor)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    capture_end = time.perf_counter()
                    capture_time = (capture_end - capture_start) * 1000

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

def consumer(frame_queue, result_queue, running):
    """
    Processes frames by performing template matching to locate the region,
    extracting three ROIs using given offsets, stacking them, and running OCR.
    OCR results and the merged ROI image are then sent via the result_queue.
    """
    try:
        ocr = init_ocr()
        template_found = False
        template_coordinates = None

        # Load the template image; ensure 'template.png' exists in your working directory
        template = cv2.imread("template.png")
        if template is None:
            logging.error("Template image not found. Please provide 'template.png'.")
            return

        # Define offsets for ROI extraction: (x_offset, y_offset, width, height)
        # Adjust these offsets to match your actual screen/template layout.
        ticker_symbol_offset = (-460, -75, 62, 18)
        qty_value_offset = (-120, -25, 62, 18)
        pos_flat_offset = (-50, 50, 62, 18)

        logging.info("Consumer started")

        while running.value:  # Check the running flag
            try:
                if not frame_queue.empty():
                    frame, capture_time, timestamp = frame_queue.get_nowait()
                    #logging.debug("Consumer: Frame retrieved from queue") #Enable for debugging queue

                    screen_np = frame

                    # If template hasn't been found yet, try to locate it via template matching
                    if not template_found:
                        result_template = cv2.matchTemplate(screen_np, template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, template_coordinates = cv2.minMaxLoc(result_template)
                        if max_val < 0.8:
                            logging.info("Template not found on screen.")
                            # Send a "not found" result (and a blank image) to the GUI
                            blank_image = np.zeros((270, 310, 3), dtype=np.uint8)
                            result_queue.put((time.time(),
                                              ["Not Found", "Not Found", "Not Found"],
                                              capture_time, 0, 0, blank_image))
                            #time.sleep(1) # Remove this sleep, it slows down the consumer
                            continue
                        else:
                            template_found = True
                            logging.info(f"Template found at: {template_coordinates}")

                    x, y = template_coordinates

                    # Helper function to crop an ROI given an offset
                    def extract_roi(offset):
                        x1, y1, w, h = offset
                        roi = screen_np[y + y1: y + y1 + h, x + x1: x + x1 + w]
                        roi = cv2.resize(roi, (310, 90))
                        return roi

                    ticker_roi = extract_roi(ticker_symbol_offset)
                    qty_roi = extract_roi(qty_value_offset)
                    pos_flat_roi = extract_roi(pos_flat_offset)

                    # Merge the three ROIs vertically (total height 90*3 = 270)
                    merged_roi = np.vstack([ticker_roi, qty_roi, pos_flat_roi])

                    # Run OCR on the merged ROI and measure processing time
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

                    ticker_text = ' '.join(rows[0]) if rows[0] else "No text"
                    qty_text = re.sub(r'\D', '', ' '.join(rows[1]) if rows[1] else "0")
                    pos_flat_text = re.sub(r'\D', '', ' '.join(rows[2]) if rows[2] else "0")
                    ocr_result = [ticker_text, qty_text, pos_flat_text]

                    total_latency = (time.perf_counter() - timestamp) * 1000

                    # Send the OCR result along with timing data and the merged ROI image
                    result_queue.put((time.time(), ocr_result, capture_time, processing_time, total_latency, merged_roi))
                    #logging.debug("Consumer: Result put in queue") #Enable for debugging queue

                else:
                    #Queue is empty
                    time.sleep(0.001) #Don't busy-wait if the queue is empty

            except mp.queues.Empty:
                #This is expected if the queue is empty during shutdown.
                logging.debug("Consumer: Frame queue empty, continuing") #Removed the print
                continue #Continue to check running.value

            except Exception as e:
                logging.error(f"Consumer error during processing: {e}")
                time.sleep(0.1) #Avoid busy-looping on error

        logging.info("Consumer stopped")
    except Exception as e:
        logging.critical(f"Consumer failed to initialize: {e}")

class UpdateThread(QThread):
    """
    A QThread subclass to handle updating the GUI with results from the result_queue.
    This allows updates to happen without blocking the main GUI thread.
    """
    result_signal = pyqtSignal(tuple)  # Define a signal to emit results

    def __init__(self, result_queue):
        super().__init__()
        self.result_queue = result_queue
        self.running = True

    def run(self):
        """Continuously check the result_queue and emit signals with new data."""
        logging.info("UpdateThread started")
        while self.running:
            try:
                if not self.result_queue.empty():
                    data = self.result_queue.get_nowait() # Added timeout
                    self.result_signal.emit(data)  # Emit the signal with the data
                    #logging.debug("UpdateThread: Data emitted") #Enable for debugging queue
                else:
                    time.sleep(0.001)  # Sleep for 1ms to avoid busy-waiting
            except mp.queues.Empty:
                #Expected during shutdown
                logging.debug("UpdateThread: Result queue empty, continuing")
                continue
            except Exception as e:
                logging.error(f"UpdateThread error: {e}")
                time.sleep(0.1) #Avoid busy-looping on error

        logging.info("UpdateThread stopped")

    def stop(self):
        """Stop the thread's event loop."""
        self.running = False
        self.quit()
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self, frame_queue, result_queue, running):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.running = running  # Shared running flag
        self.prod_process = None  # Initialize as None
        self.consumer_process = None  # Initialize as None
        self.update_thread = None # Initialize as None
        self.setWindowTitle("OCR Results")
        self.resize(400, 500)  # Increased height to accommodate buttons

        # Create labels to display OCR text results
        self.ticker_label = QLabel("Ticker: ")
        self.qty_label = QLabel("Quantity: ")
        self.pos_label = QLabel("Position/Flat: ")

        # Label to display the merged ROI image (visual feedback)
        self.image_label = QLabel()
        self.image_label.setFixedSize(310, 270)

        # Labels for displaying timing information
        self.capture_time_label = QLabel("Capture Time: ")
        self.ocr_time_label = QLabel("OCR Time: ")
        self.total_latency_label = QLabel("Total Latency: ")

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

        # Create Buttons
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.exit_button = QPushButton("Exit")

        self.start_button.clicked.connect(self.start_processes)
        self.stop_button.clicked.connect(self.stop_processes)
        self.exit_button.clicked.connect(self.close)  # Use built in close method

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        # button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.exit_button)

        # Overall layout
        layout = QVBoxLayout()
        layout.addWidget(self.ticker_label)
        layout.addWidget(self.qty_label)
        layout.addWidget(self.pos_label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.capture_time_label)
        layout.addWidget(self.ocr_time_label)
        layout.addWidget(self.total_latency_label)
        layout.addLayout(divider_layout)
        layout.addLayout(button_layout)  # Add button layout

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Disable stop button at start
        self.stop_button.setEnabled(False)

    @asyncSlot(tuple)
    async def handle_update(self, data):
        await self.update_gui(data)

    async def update_gui(self, data):
        """Updates GUI elements with new OCR data."""
        try:
            if len(data) == 6:
                timestamp, result, capture_time, ocr_time, total_latency, merged_roi = data
                ticker_text, qty_text, pos_text = result

                # Update the text labels
                self.ticker_label.setText(f"Ticker: {ticker_text}")
                self.qty_label.setText(f"Quantity: {qty_text}")
                self.pos_label.setText(f"Position/Flat: {pos_text}")

                # await place_alpaca_order(ticker_text, qty_text, "BUY")
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

                # Get the action based on the tuple (a, b, c)
                if qty_text != "4500":
                    action_key = ("not_4500", pos_text)
                else:
                    action_key = (qty_text, pos_text)
                action = cases.get(action_key, lambda: print("No match"))
                action()

                # Convert merged_roi (a NumPy array in BGR) to QImage and update the image label
                height, width, channel = merged_roi.shape
                bytesPerLine = 3 * width
                qimg = QImage(merged_roi.data, width, height, bytesPerLine, QImage.Format.Format_BGR888)
                pixmap = QPixmap.fromImage(qimg)
                self.image_label.setPixmap(pixmap)

                # Update the timing labels
                self.capture_time_label.setText(f"Capture Time: {capture_time:.2f} ms")
                self.ocr_time_label.setText(f"OCR Time: {ocr_time:.2f} ms")
                self.total_latency_label.setText(f"Total Latency: {total_latency:.2f} ms")
            else:
                logging.warning(f"update_gui: Unexpected data format: {data}")
        except Exception as e:
            logging.error(f"Error in update_gui: {e}")

    def start_processes(self):
        """Starts the producer, consumer, and update thread."""
        # Enable stop button, disable start
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.running.value = True  # Set running to true
        logging.info("Starting processes...")

        # Create new process objects
        self.prod_process = mp.Process(target=producer, args=(self.frame_queue, self.running)) #You can add a monitor number here
        self.consumer_process = mp.Process(target=consumer, args=(self.frame_queue, self.result_queue, self.running))
        self.update_thread = None
        self.update_thread = UpdateThread(self.result_queue) #Create a new update thread

        # Connect the signal from the update thread to the update_gui method
        self.update_thread.result_signal.connect(self.handle_update)

        # Start the processes and thread
        self.prod_process.start()
        self.consumer_process.start()
        self.update_thread.start()
        logging.info("Processes started") 

    def stop_processes(self):
        """Stops the producer, consumer, and update thread."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        logging.info("Stopping processes...")
        self.running.value = False  # Signal processes to stop

        # Stop update thread
        if self.update_thread:
            self.update_thread.stop()
            self.update_thread.terminate()
            self.update_thread = None

        # Gracefully stop producer and consumer
        if self.prod_process:
            self.prod_process.join(timeout=2)
            if self.prod_process.is_alive():
                self.prod_process.terminate()
            self.prod_process = None

        if self.consumer_process:
            self.consumer_process.join(timeout=2)
            if self.consumer_process.is_alive():
                self.consumer_process.terminate()
            self.consumer_process = None

        # Clear queues to remove any residual data
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
        """Handle the closing of the main window."""
        self.stop_processes()  # Stop processes before closing
        super().closeEvent(event)

if __name__ == '__main__':
    # Use 'spawn' for multiprocessing compatibility (especially on Windows)
    try:
        mp.set_start_method("spawn")

        frame_queue = mp.Queue(maxsize=5)  # Small queue for near real-time performance
        result_queue = mp.Queue()

        # Shared running flag
        running = mp.Value('b', False)  # Boolean value, initially false

        # Initialize and show the PyQt6 GUI
        app = QApplication(sys.argv)
        import qasync
        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)

        with loop:
            main_window = MainWindow(frame_queue, result_queue, running)  # Pass queues and running flag
            main_window.show()
            loop.run_forever()
    finally:
        loop.close()
