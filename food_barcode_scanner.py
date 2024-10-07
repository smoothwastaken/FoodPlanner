import time
from typing import List, Optional

import cv2
import openfoodfacts
from PIL import Image
from pyzbar import pyzbar


class FoodProduct:
    """
    Represents a food product with information retrieved from OpenFoodFacts.
    """

    def __init__(self, barcode: str):
        """
        Initialize a FoodProduct instance.

        Args:
            barcode (str): The barcode of the food product.
        """
        self.barcode: str = barcode
        self.name: str = "Not available"
        self.brand: str = "Not available"
        self.ingredients: str = "Not available"

    def load_info(self, api: openfoodfacts.API) -> bool:
        """
        Load product information from OpenFoodFacts API.

        Args:
            api (openfoodfacts.API): The OpenFoodFacts API instance.

        Returns:
            bool: True if product information was successfully loaded, False otherwise.
        """
        product = api.product.get(
            self.barcode, fields=["product_name", "brands", "ingredients_text"]
        )
        if product:
            self.name = product.get("product_name", self.name)
            self.brand = product.get("brands", self.brand)
            self.ingredients = product.get("ingredients_text", self.ingredients)
            return True
        return False

    def __str__(self) -> str:
        """
        Return a string representation of the FoodProduct.

        Returns:
            str: A formatted string containing product information.
        """
        return f"Barcode: {self.barcode}\nName: {self.name}\nBrand: {self.brand}\nIngredients: {self.ingredients}"


class BarcodeScanner:
    """
    A class to scan barcodes from images and retrieve product information.
    """

    def __init__(self):
        """
        Initialize a BarcodeScanner instance.
        """
        self.api: openfoodfacts.API = openfoodfacts.API(user_agent="MyAppScanner/1.0")

    def scan_image(self, image_path: str) -> List[FoodProduct]:
        """
        Scan an image for barcodes and retrieve product information.

        Args:
            image_path (str): The path to the image file to be scanned.

        Returns:
            List[FoodProduct]: A list of FoodProduct instances for the scanned products.
        """
        image: Image.Image = Image.open(image_path)
        codes: List[pyzbar.Decoded] = pyzbar.decode(image)
        products: List[FoodProduct] = []

        for code in codes:
            barcode: str = code.data.decode("utf-8")
            product: FoodProduct = FoodProduct(barcode)
            if product.load_info(self.api):
                products.append(product)
            else:
                print(f"Product not found for barcode: {barcode}")

        return products


class ContinuousProductScanner:
    """
    A class for continuous scanning of product barcodes using OpenCV and the camera.
    """

    def __init__(self, camera_id: int = 0):
        """
        Initialize the ContinuousProductScanner.

        Args:
            camera_id (int): The ID of the camera to use. Defaults to 0.
        """
        self.camera_id: int = camera_id
        self.cap = cv2.VideoCapture = cv2.VideoCapture(self.camera_id)
        self.last_scan_time: float = 0
        self.scan_interval: float = 2.0  # Minimum time between scans in seconds
        self.barcode_scanner: BarcodeScanner = BarcodeScanner()

    def scan_products(self) -> None:
        """
        Continuously scan for product barcodes using the camera.
        """
        while True:
            # Capture frame-by-frame with custom frame-rate
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            barcodes: List[pyzbar.Decoded] = self.detect_barcodes(frame)
            self.process_barcodes(frame, barcodes)

            cv2.imshow("Product Scanner", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def detect_barcodes(self, frame: cv2.Mat) -> List[pyzbar.Decoded]:
        """
        Detect barcodes in the given frame.

        Args:
            frame (cv2.Mat): The image frame to process.

        Returns:
            List[pyzbar.Decoded]: A list of detected barcodes.
        """
        return pyzbar.decode(frame)

    def process_barcodes(self, frame: cv2.Mat, barcodes: List[pyzbar.Decoded]) -> None:
        """
        Process detected barcodes and draw them on the frame.

        Args:
            frame (cv2.Mat): The image frame to draw on.
            barcodes (List[pyzbar.Decoded]): The list of detected barcodes.
        """
        for barcode in barcodes:
            barcode_data: str = barcode.data.decode("utf-8")
            barcode_type: str = barcode.type

            # Draw bounding box
            (x, y, w, h) = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Put barcode data and type on the image
            text: str = f"{barcode_data} ({barcode_type})"
            cv2.putText(
                frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
            )

            # Print barcode data and product info if enough time has passed since last scan
            current_time: float = time.time()
            if current_time - self.last_scan_time >= self.scan_interval:
                print(f"Scanned: {barcode_data} ({barcode_type})")
                product: FoodProduct = FoodProduct(barcode_data)
                if product.load_info(self.barcode_scanner.api):
                    print(product)
                else:
                    print("Product information not found")
                self.last_scan_time = current_time


def main() -> None:
    """
    Main function to demonstrate the usage of the ContinuousProductScanner.
    """
    # Example usage of ContinuousProductScanner
    print("\nStarting continuous scanner. Press 'q' to quit.")
    continuous_scanner: ContinuousProductScanner = ContinuousProductScanner()
    continuous_scanner.scan_products()


if __name__ == "__main__":
    main()
