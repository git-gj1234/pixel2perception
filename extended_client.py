import cv2
import speech_recognition as sr
import pyttsx3
import subprocess
import time
import threading
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Label, Button
import google.generativeai as genai
import easyocr
import numpy as np
from transformers import pipeline
import random

history_questions = ["", "", "", "", ""]
history_answers = ["", "", "", "", ""]

print("script start")
pipe = pipeline("image-segmentation", model="badmatr11x/semantic-image-segmentation")
print("OCR model loaded")

default_dir = r"C:\Users\adith\Documents\bmsce\3rd year\6th SEM\mini project\eye for the blind client"
response_path = r"C:\Users\adith\Documents\bmsce\3rd year\6th SEM\mini project\eye for the blind client\git_response_prompt\file.txt"

def run_git_pull():
    subprocess.run(['python', 'git_pull.py'])

git_pull_thread = threading.Thread(target=run_git_pull)
git_pull_thread.start()


def check_commit_push(repo_path, commit_message="New image to process"):
    os.chdir(repo_path)
    # git_status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    # if git_status.stdout:
    # print("There are local changes.")
    subprocess.run(['git', 'add', '.'])
    subprocess.run(['git', 'commit', '-m', commit_message])
    subprocess.run(['git', 'push', 'origin', 'main'])
    print("Changes committed and pushed to remote.")
    # else:
    #     print("There are no local changes.")

def clear_and_write_file(file_path, new_content):
    try:
        # Open the file in write mode, which clears existing content
        with open(file_path, 'w') as file:
            # Write the new content to the file
            file.write(new_content)
        print(f"Successfully cleared and wrote to '{file_path}'.")
    except IOError:
        print(f"Error: Failed to write to file '{file_path}'.")


def get_modification_time(file_path):
    return os.path.getmtime(file_path)

def read_text_file(file_path):
    file_content = None
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
            # print(str(file_content))
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except IOError:
        print(f"Error: Failed to read from file '{file_path}'.")
    return str(file_content)

check_change = read_text_file(r"git_response_prompt/file.txt")

def remove_unnecessary(string):
    print(string)
    input_string = string
    index = input_string.find('\\nAssistant:') + len('\nAssistant:')+1
    output_string = input_string[index:].strip()
    output_string = output_string[:-3]
    print(output_string)
    return output_string

def function_model(image, text, lbl_output):
    global check_change
    check_change = read_text_file(r"git_response_prompt/file.txt")
    os.chdir(default_dir)
    cv2.imwrite(r"C:\Users\adith\Documents\bmsce\3rd year\6th SEM\mini project\eye for the blind client\git_image\image.jpg", image)
    clear_and_write_file(r"C:\Users\adith\Documents\bmsce\3rd year\6th SEM\mini project\eye for the blind client\git_image\question.txt", str(text))
    check_commit_push(r"C:\Users\adith\Documents\bmsce\3rd year\6th SEM\mini project\eye for the blind client\git_image")
    os.chdir(default_dir)
    while True:
        try:
            #print("check:", check_change)

            new_str = read_text_file(r"git_response_prompt/file.txt")
            #print("new: ", new_str)
            if new_str != check_change:
                check_change = new_str
                break
        except Exception as e:
            pass

    print("Attempting file read")
    time.sleep(1)
    with open(response_path, 'r') as file2:
        file_contents = file2.read()
    # print(file_contents)
    file_contents = remove_unnecessary(str(file_contents))
    # print("After removel:",file_contents)
    lbl_output.config(text=file_contents)
    print(file_contents)
    return file_contents

def generate_for_ocr(input_text, prompt):
    os.environ["AIzaSyDmqBXVPGMFO7OQHquImxkAlvUbyqAzdDo"] = (
        "AIzaSyDmqBXVPGMFO7OQHquImxkAlvUbyqAzdDo"
    )
    genai.configure(api_key=os.environ["AIzaSyDmqBXVPGMFO7OQHquImxkAlvUbyqAzdDo"])
    model = genai.GenerativeModel(model_name="gemini-1.0-pro")
    inp = f"""describe this text recognized from OCR for blind person. 
    
    OCR Text:
    {input_text}

    
    Instructions for Task Completion:
    - Your output should be a detailed description of the provided text for a blind person to understand.
    - Focus on capturing the essence and key details of the text.
    - Easy english.
    - Ignore random letters and symbols 
    - tell only about the main thing
    Answer the following question using all the info given to u
    {prompt}
    """
    response = model.generate_content([inp])
    print(response.text)
    return response.text


def recognize_text(image):
    reader = easyocr.Reader(["en"], gpu=True)
    text_detections = reader.readtext(image)
    recognized_texts = []

    if text_detections:
        for bbox, text, score in text_detections:
            if score > 0.25:
                recognized_texts.append(text)
    else:
        print("No text detected")

    if recognized_texts:
        return ", ".join(recognized_texts)
    else:
        return "No text detected"

def get_object_of_intrest(data, prompt):
    os.environ["AIzaSyDmqBXVPGMFO7OQHquImxkAlvUbyqAzdDo"] = (
        "AIzaSyDmqBXVPGMFO7OQHquImxkAlvUbyqAzdDo"
    )
    genai.configure(api_key=os.environ["AIzaSyDmqBXVPGMFO7OQHquImxkAlvUbyqAzdDo"])
    model = genai.GenerativeModel(model_name="gemini-1.0-pro")
    inp = f"""
    We are building a vision language model. You are to act as the interface.
    We used a semantic segmentation model to get objects and areas.
    These will be given to you as data: {data}
    Find the object with the closest reference to the object inquired
    about by the user. Use data of both names and objects
    areas to decide which is the most feasible object which will usually
    be the object with the biggest area
    referenced by the user and print only the object as is, no more no less.
    User prompt: {prompt}
    """

    # Generate the response from the model
    response = model.generate_content([inp])

    # Print the response
    return response.text

def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def get_largest_contour(binary_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(
        cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def return_bounding_boxes_labels(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pipe(r"captured_image.jpg")
    overlay = np.zeros_like(image_rgb)
    
    min_area_threshold = 1000

    bounding_boxes = []
    labels = []

    for result in results:
        mask = result["mask"]
        label = result["label"]
        color = get_random_color()
        mask = np.array(mask)
        binary_mask = np.where(mask > 0, 1, 0).astype(np.uint8)
        largest_contour = get_largest_contour(binary_mask)
        if largest_contour is not None:
            if cv2.contourArea(largest_contour) >= min_area_threshold:
                colored_mask = np.zeros_like(image_rgb)
                cv2.drawContours(
                    colored_mask, [largest_contour], -1, color, thickness=cv2.FILLED
                )
                overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)
                x, y, w, h = cv2.boundingRect(largest_contour)
                bounding_boxes.append((x, y, w, h))
                labels.append(label)

    combined = cv2.addWeighted(image_rgb, 0.2, overlay, 0.8, 0)

    for (x, y, w, h), label in zip(bounding_boxes, labels):
        cv2.rectangle(combined, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(
            combined, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2
        )
    output_string = ", ".join(
    [f"{label} area {w * h}" for (x, y, w, h), label in zip(bounding_boxes, labels)]
    )
    return output_string, bounding_boxes, labels

def ocr_funct(prompt, imghat):
    output_string, bounding_boxes, labels = return_bounding_boxes_labels(imghat)
    index = get_object_of_intrest(output_string, prompt)
    index = index.lower()
    # print("LABEL : ", index)
    i = labels.index(index)
    (x, y, w, h) = bounding_boxes[i]
    imghat = imghat[x : x + w, y : y + h]
    final_ocr = recognize_text(imghat)
    print("ITS WRITTEN :", final_ocr)
    return final_ocr


def decision(frame, text, lbl_output, history_questions, history_answers):
    prompt = text
    os.environ["AIzaSyDmqBXVPGMFO7OQHquImxkAlvUbyqAzdDo"] = (
        "AIzaSyDmqBXVPGMFO7OQHquImxkAlvUbyqAzdDo"
    )
    genai.configure(api_key=os.environ["AIzaSyDmqBXVPGMFO7OQHquImxkAlvUbyqAzdDo"])
    model = genai.GenerativeModel(model_name="gemini-1.0-pro")
    inp = f"""
    We are building a vision-language model. You will act as the interface for this model.
    A user will ask a question, and you must decide the appropriate action based on the question:
    - Say "0" if the prompt requires OCR (Optical Character Recognition).
    - Say "1" if the prompt requires the use of the VQA (Visual Question Answering) model.
    - If the question can be answered using existing knowledge, respond with the answer directly. This existing knowledge is available as two arrays of strings, each containing the last 5 questions and answers.
      Note that In many cases the user might say explicilty to use previous info.

    Remember:
    - Use "0" for OCR when specific text needs to be read.
    - Use "1" for VQA to act as the user's eyes for general visual information.
    - Answer directly if the question matches previous knowledge. If the answer is found in the previous knowledge, respond directly without saying "2".

    When answering directly:
    - (this is an exampleu wont always read a menu)Determine object references intelligently. For example, if a question about a menu was asked five questions ago and the new question is also about a menu, use that previous answer if it applies.

    Here is the previous information:
    - Previous questions: {history_questions}
    - Previous answers: {history_answers}

    The prompt is: {prompt}
    """
    response = model.generate_content([inp])
    # return response.text
    val = response.text
    ques = text
    print(val)
    if val == '0': 
        image = cv2.imread(r"C:\Users\adith\Documents\bmsce\3rd year\6th SEM\mini project\eye for the blind client\captured_image.jpg")
        val2 = (ocr_funct(prompt, image))
        val3 = generate_for_ocr(val2, {"Whats on the menu? Give a summary"})
        history_questions.pop(0)
        history_answers.pop(0)
        history_questions.append(ques)
        history_answers.append(val3)
    elif len(val)>2:
        print(val)
        history_questions.pop(0)
        history_answers.pop(0)
        history_questions.append(ques)
        history_answers.append(val)
    else:
        val2 = function_model(frame, text, lbl_output)
        history_questions.pop(0)
        history_answers.pop(0)
        history_questions.append(ques)
        history_answers.append(val2)
    return str(history_answers[-1])



def speech_to_text(lbl_output):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        lbl_output.config(text="Can you describe the appetizer section")
        audio = recognizer.listen(source)
    # print(1)
    try:
        lbl_output.config(text="Recording...")
        text = recognizer.recognize_google(audio, language="en-US")
    except sr.UnknownValueError:
        lbl_output.config(text="Google Web Speech API could not understand the audio")
        text = ""
    except sr.RequestError as e:
        lbl_output.config(
            text=f"Could not request results from Google Web Speech API; {e}"
        )
        text = ""
    # print(text)
    return text


def text_to_speech(text, voice_name="David", lbl_output=None):
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    voice_id = None
    for voice in voices:
        if voice_name.lower() in voice.name.lower():
            voice_id = voice.id
            break
    if voice_id is None:
        default_message = f"Voice '{voice_name}' not found. Using default voice."
        print(default_message)
        if lbl_output:
            lbl_output.config(text=default_message)
        voice_id = voices[0].id
    engine.setProperty("voice", voice_id)
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1)
    engine.say(text)
    engine.runAndWait()


def update_frame():
    global lbl, cap
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk = imgtk
        lbl.configure(image=imgtk)
    lbl.after(10, update_frame)


def capture_image():
    global lbl_output
    capture_thread = threading.Thread(target=capture_image_thread)
    capture_thread.start()


def capture_image_thread():
    global lbl_output, cap
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("captured_image.jpg", frame)
        lbl_output.config(text="Image captured!")
        text = speech_to_text(lbl_output)
        lbl_output.config(text=f"You said: {text}")
        text_to_speech(text=f"You said: {text}", lbl_output=lbl_output)
        reply = decision(frame, text, lbl_output, history_questions, history_answers)
        text_to_speech(reply, lbl_output=lbl_output)


def quit_app():
    global root, cap
    root.quit()
    cap.release()
    cv2.destroyAllWindows()


background_color = "#f2f2f2"
button_color = "#d1ffbd"
button_hover_color = "#b39ddb"
text_color = "#333"
button_style = {
    "bg": button_color,
    "fg": "black",
    "font": ("LuckiestGuy", 12),
    "activebackground": button_hover_color,
    "activeforeground": "black",
    "border": 2,
    "relief": "solid",
    "padx": 10,
    "pady": 5,
}
button_width = 15
button_height = 2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Pixel2Perception")
root.geometry("800x800")

# Frame for title and image
frame_title = tk.Frame(root, bg=background_color)
frame_title.pack(pady=(20, 10), fill=tk.X)

# Load and resize image
img = Image.open("blind.png")
img = img.resize((50, 50), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)

# Label for image
lbl_image = Label(frame_title, image=img, bg=background_color)
lbl_image.image = img  # Keep a reference to the image to prevent garbage collection
lbl_image.grid(row=0, column=0, padx=(5, 5))

# Label for title
lbl_title = Label(
    frame_title,
    text="Pixel2Perception",
    font=("LuckiestGuy", 20, "bold"),  # Luckiest Guy font
    anchor="center",
    fg=text_color,
    bg=background_color,
)
lbl_title.grid(row=0, column=1)

# Label for camera feed
lbl = Label(root)
lbl.pack()

# Capture button
btn_capture = Button(
    root,
    text="Ask",
    command=capture_image,
    width=button_width,
    height=button_height,
    **button_style,
)
btn_capture.pack(pady=20)

# Output label
lbl_output = Label(root, text="", fg=text_color)
lbl_output.pack(pady=10)

# Start camera update thread
update_thread = threading.Thread(target=update_frame)
update_thread.start()

root.mainloop()
