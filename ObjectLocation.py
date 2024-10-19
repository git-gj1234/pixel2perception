import os
import requests
from groq import Groq
from dotenv import load_dotenv
from PIL import Image
import base64


class ObjectDetectionAssistant:
    def __init__(
        self,
        hf_api_key,
        groq_api_key,
        api_url="https://api-inference.huggingface.co/models/hustvl/yolos-small",
    ):
        """Initialize the object detection assistant with API keys and endpoint URLs."""
        self.hf_api_key = hf_api_key
        self.groq_api_key = groq_api_key
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {hf_api_key}"}
        self.client = Groq(api_key=groq_api_key)

    def query_image(self, filename):
        """Uploads an image to the YOLOS model and returns the detected objects."""
        with open(filename, "rb") as f:
            data = f.read()
        response = requests.post(self.api_url, headers=self.headers, data=data)

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")

        return response.json()

    def determine_position(self, box, img_width, img_height):
        """Determine the position of the object based on bounding box coordinates."""
        xmin = float(box["xmin"])
        ymin = float(box["ymin"])
        xmax = float(box["xmax"])
        ymax = float(box["ymax"])

        # Calculate the center of the bounding box
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2

        # Define thresholds for dividing the image into 9 zones
        x_left_threshold = img_width * 0.33
        x_right_threshold = img_width * 0.66
        y_top_threshold = img_height * 0.33
        y_bottom_threshold = img_height * 0.66

        # Classify the bounding box based on its center position
        if x_center < x_left_threshold and y_center < y_top_threshold:
            return "top-left"
        elif x_center > x_right_threshold and y_center < y_top_threshold:
            return "top-right"
        elif x_center < x_left_threshold and y_center > y_bottom_threshold:
            return "bottom-left"
        elif x_center > x_right_threshold and y_center > y_bottom_threshold:
            return "bottom-right"
        elif (
            x_left_threshold <= x_center <= x_right_threshold
            and y_center < y_top_threshold
        ):
            return "top-center"
        elif (
            x_left_threshold <= x_center <= x_right_threshold
            and y_center > y_bottom_threshold
        ):
            return "bottom-center"
        elif (
            x_center < x_left_threshold
            and y_top_threshold <= y_center <= y_bottom_threshold
        ):
            return "left-center"
        elif (
            x_center > x_right_threshold
            and y_top_threshold <= y_center <= y_bottom_threshold
        ):
            return "right-center"
        else:
            return "center"

    def format_detection_results(self, output, img_width, img_height):
        """Format the detection results into a readable string."""
        if not isinstance(output, list):
            raise ValueError(
                f"Expected a list of detected objects but got: {type(output)}"
            )

        detection_results = ""
        for obj in output:
            label = obj.get("label", "unknown")
            box = obj.get("box", {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0})
            position = self.determine_position(box, img_width, img_height)

            xmin = box.get("xmin", 0)
            xmax = box.get("xmax", 0)
            ymin = box.get("ymin", 0)
            ymax = box.get("ymax", 0)

            detection_results += (
                f"Label: {label}\n"
                f"Position: {position}\n"
                f"Xcoords: ({xmin}, {xmax})\n"
                f"Ycoords: ({ymin}, {ymax})\n\n"
            )

        return detection_results

    def encode_image(self, image_path):
        """Encode the image in base64 format."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def describe_image(self, image_path):
        """Use the Groq API to describe the image."""
        base64_image = self.encode_image(image_path)
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            model="llama-3.2-90b-vision-preview",
        )

        return chat_completion.choices[0].message.content

    def assist_user(self, image_path, user_prompt):
        """Use the Groq API to assist the user with spatial guidance based on detected objects."""
        output = self.query_image(image_path)
        img = Image.open(image_path)
        img_width, img_height = img.size
        detection_results = self.format_detection_results(output, img_width, img_height)
        desc = self.describe_image(image_path)

        system_prompt = """
Act like an expert assistant specialized in guiding visually impaired users to locate objects in their environment. 
You have extensive experience in using object detection data to provide spatially-aware, clear, and context-rich 
spoken instructions that are practical for everyday use. Your responses should be natural, user-friendly, and always 
oriented toward improving the user's spatial understanding of their surroundings.

Objective: Your primary task is to analyze object detection results and convert them into detailed, spoken instructions 
to help the user locate objects or understand the arrangement of items in their immediate environment.

Context: You will receive object detection results in a list format, where each item contains:
- score: Detection confidence (in percentage or a decimal),
- label: The name of the detected object,
- box: Bounding box coordinates (xmin, ymin, xmax, ymax) indicating the objects location within the frame.

The user may ask questions like “Where is my [object]?” or “What objects are around me?” You will interpret the detection results, 
analyzing bounding box data, and provide intuitive spatial guidance. Ensure that your responses are sensitive to the users 
needs and include relevant details without overwhelming them.

Response Strategy:
- Focus on the Users Query: If the user asks about a specific object, prioritize its detection and location in your response. 
Clearly state its spatial position relative to the user. If the object is not detected, let the user know that it is not in view and 
suggest possible next steps (e.g., repositioning the camera).

- Spatial Orientation: Use bounding box coordinates to describe the object's position relative to the user:

If xmin is near the left boundary of the frame, describe it as “to your left.”
If xmax is near the right boundary, describe it as “to your right.”
If both xmin and xmax are centered, describe the object as “in front of you” or “directly ahead.”
For vertical positioning, use ymin and ymax to describe objects as being "low," "mid-level," or "high."

Multiple Objects of the Same Type: If more than one instance of the same object type is detected, differentiate them by their positions. 
For example, "There are two chairs: one to your left and one directly in front of you."


When possible, use recognized objects like “tables” or “chairs” as landmarks to help users orient themselves. 
For example, “Your book is on the table in front of you, slightly to the right.”

If an object the user asks for is not detected, explain its absence clearly and suggest actions such as 
repositioning the camera or scanning the environment again.

Always communicate in a friendly and clear manner, aiming to build trust and reduce any potential 
frustration. Use guiding phrases such as “It might help to adjust your camera slightly upward” if needed.

- Make sure to not talk about coordinates, that will be of no help to the user.
- Use the surrounding description to help in locating the mentioned object.

        """
        message_content = f" This is the description of the surroundings: {desc}. Detected objects: {detection_results}. User query: {user_prompt}."

        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message_content},
            ],
            model="llama3-groq-70b-8192-tool-use-preview",
        )

        return chat_completion.choices[0].message.content


# Example usage
if __name__ == "__main__":
    load_dotenv()
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

    assistant = ObjectDetectionAssistant(hf_api_key, groq_api_key)

    img_pth = "1.jpg"
    user_query = "where is my laptop?"

    result = assistant.assist_user(img_pth, user_query)
    print(result)
