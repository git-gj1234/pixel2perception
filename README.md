"""
# Pixel2Perception

Pixel2Perception is an innovative system designed to empower visually impaired users by helping them understand and navigate their environment through real-time image analysis and object detection. By leveraging cutting-edge AI technologies, the system provides spoken instructions about the location and arrangement of objects in the user's surroundings, offering valuable assistance in everyday tasks.

## Features

- **Object Detection**: Integrates a YOLO model to detect various objects within an image, enabling users to identify items in their environment.
- **Spatial Analysis**: Determines the relative position of detected objects, offering a spatial understanding of their layout.
- **Image Description**: Utilizes advanced vision AI models to generate rich, detailed descriptions of the surroundings.
- **User Assistance**: Provides auditory instructions based on image analysis, giving users actionable information to help them locate and interact with objects.

---

## Files Overview

- **Barcode.py**  
  This script is responsible for scanning barcodes and identifying products or objects based on the barcode data. It enhances the system’s ability to detect labeled items.

- **M blind.png**  
  A resource file, possibly an image used in the interface or as a placeholder for testing.

- **captured_image.jpg**  
  A sample image used for testing the object detection system.

- **client_server_connector.py**  
  Handles the communication between the client and server. This script ensures that the data generated from the user’s environment is transmitted to the server for analysis and instructions are sent back to the client.

- **Main.py**  
  The main interface script that coordinates the entire system. This file brings together all the functionalities—image capture, object detection, and spoken instructions—providing the core user interaction.

- **ObjectLocation.py**  
  Focuses on the spatial analysis of detected objects, determining their position relative to the user’s perspective. This data is crucial for providing accurate spoken instructions about the location of objects.

- **Prompts.py**  
  Contains all the prompts sent to language models (LLMs) for generating detailed descriptions or instructions based on object detection and spatial data. This enhances the system’s ability to communicate effectively with the user.

- **server_code.ipynb**  
  The backend server code is hosted in this Jupyter notebook. It processes the images, detects objects, performs spatial analysis, and sends the results back to the client. This is the core processing unit for Pixel2Perception.

---

## How It Works

1. The user captures an image of their environment using the system interface.
2. The image is sent to the backend server for processing.
3. The YOLO model detects objects, and the system performs spatial analysis to determine their position relative to the user.
4. The system generates a detailed description of the environment, including the location and identity of objects.
5. Spoken instructions are provided to the user, guiding them on how to interact with or locate objects.

---

## Installation

1. Clone this repository.
2. Install the required dependencies listed in the `requirements.txt`.
3. Set up the environment variables by creating a `.env` file.
4. Run the backend server by executing the Jupyter notebook `server_code.ipynb`.
5. Launch the main interface using `Main.py`.

---

## Future Work

- **Enhanced Object Recognition**: Integrate more advanced models to recognize a broader array of objects.
- **Mobile Support**: Expand functionality to mobile platforms for portability.
- **Real-Time Processing**: Optimize for faster image capture and analysis, providing real-time feedback.
"""
