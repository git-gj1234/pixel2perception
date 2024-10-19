import cv2
from pyzbar import pyzbar
import urllib.request
import json
import pprint

class BarcodeDecoder:
    def __init__(self, image_file):
        self.image_file = image_file
        self.img = cv2.imread(self.image_file)
        self.barcode = None
        self.barcode_type = None

    def decode(self):
        if self.img is None:
            print(f"Error: Unable to load image {self.image_file}. Check the file path.")
            return None

        decoded_objects = pyzbar.decode(self.img)
        if not decoded_objects:
            print("No barcodes found.")
        else:
            for obj in decoded_objects:
                self.img = self.draw_barcode(obj)
                self.barcode = obj.data.decode('utf-8')
                self.barcode_type = obj.type
                print("Type:", obj.type)
                print("Data:", self.barcode)
                print()

        return self.img, self.barcode, self.barcode_type

    def draw_barcode(self, decoded):
        image = cv2.rectangle(
            self.img,
            (decoded.rect.left, decoded.rect.top),
            (decoded.rect.left + decoded.rect.width, decoded.rect.top + decoded.rect.height),
            color=(0, 255, 0),
            thickness=5
        )
        return image

class BarcodeLookup:
    def __init__(self, barcode, api_key):
        self.barcode = barcode
        self.api_key = api_key

    def lookup(self):
        url = f"https://api.barcodelookup.com/v3/products?barcode={self.barcode}&formatted=y&key={self.api_key}"

        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
            barcode_number = data["products"][0]["barcode_number"]
            name = data["products"][0]["title"]
            print("Barcode Number:", barcode_number)
            print("Title:", name)
            print("\nEntire Response:")
            pprint.pprint(data)
        except Exception as e:
            print(f"Error during barcode lookup: {e}")

if __name__ == "__main__":
    image_file = "qr.png"  

    decoder = BarcodeDecoder(image_file)
    img, barcode, barcode_type = decoder.decode()

    if barcode_type and barcode_type != "QRCODE":
        print("Non-QRCODE barcode found, looking up details online...")
        api_key = "w5nhj5noo3ti7rtqw1pfygdlan2zxi"
        lookup = BarcodeLookup(barcode, api_key)
        lookup.lookup()
    else:
        print("QR code detected or no valid barcode found, skipping API lookup.")
