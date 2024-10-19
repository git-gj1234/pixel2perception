import cv2
from pyzbar import pyzbar
import urllib.request
import json
import pprint
from groq import Groq
import os


# Unified BarcodeProcessor Class
class BarcodeProcessor:
    def __init__(self):
        self.api_key = "w5nhj5noo3ti7rtqw1pfygdlan2zxi"
        self.barcode = None
        self.barcode_type = None
        self.img = None
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.data = ""

    def decode(self, img):
        self.img = img
        if self.img is None:
            print(f"Error: Unable to load image {self.img }. Check the file path.")
            return None

        decoded_objects = pyzbar.decode(self.img)
        if not decoded_objects:
            print("No barcodes found.")
        else:
            for obj in decoded_objects:
                self.img = self.draw_barcode(obj)
                self.barcode = obj.data.decode("utf-8")
                self.barcode_type = obj.type
                print("Type:", obj.type)
                print("Data:", self.barcode)
                print()

        return self.img, self.barcode, self.barcode_type

    def draw_barcode(self, decoded):
        # Draw a rectangle around the detected barcode
        image = cv2.rectangle(
            self.img,
            (decoded.rect.left, decoded.rect.top),
            (
                decoded.rect.left + decoded.rect.width,
                decoded.rect.top + decoded.rect.height,
            ),
            color=(0, 255, 0),
            thickness=5,
        )
        return image

    def lookup(self):
        if not self.api_key:
            print("API key is missing. Cannot perform barcode lookup.")
            return

        if self.barcode_type and self.barcode_type != "QRCODE":
            url = f"https://api.barcodelookup.com/v3/products?barcode={self.barcode}&formatted=y&key={self.api_key}"

            try:
                with urllib.request.urlopen(url) as response:
                    data = json.loads(response.read().decode())

                # Extract and print information
                barcode_number = data["products"][0]["barcode_number"]
                name = data["products"][0]["title"]
                print("Barcode Number:", barcode_number)
                print("Title:", name)
                print("\nEntire Response:")
                pprint.pprint(data)
                self.data = data
            except Exception as e:
                print(f"Error during barcode lookup: {e}")
        else:
            print("QR code detected or no valid barcode found, skipping API lookup.")

    def barcode_llm(self, prompt):
        system_message = """
            You are a model that assists visually impaired users by interpreting and describing product information retrieved from a scanned barcode. 
            Your goal is to extract the essential details from the product data and present them clearly and concisely.
            """

        user_message = f"""
            You are tasked with describing the product information obtained from a scanned barcode. The data may contain random symbols or irrelevant characters, 
            so your job is to focus only on the key product details.

            Product Data:
            "{self.data}"

            Instructions for Task Completion:
            1. Provide a detailed description of the essential product information, such as name, price, and relevant details.
            2. Use clear, simple language to ensure comprehension by a blind person.
            3. Ignore any random symbols, stray characters, or irrelevant information.
            4. Highlight the most important points clearly and avoid unnecessary details.

            Additionally, answer the following user question based on the product information:
            "{prompt}"


            If there is not barcode detected then just say No barcode detected.
            If its detected, just give the answer related to the users questions and nothing else more.
            """

        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            model="llama-3.1-70b-versatile",
        )

        response_text = chat_completion.choices[0].message.content
        print(response_text)
        return response_text


# # Main Logic
# if __name__ == "__main__":
#     image_file = "bar.jpg"
#     api_key = "w5nhj5noo3ti7rtqw1pfygdlan2zxi"

#     processor = BarcodeProcessor(image_file, api_key)
#     img, barcode, barcode_type = processor.decode()

#     if barcode:
#         processor.lookup()
