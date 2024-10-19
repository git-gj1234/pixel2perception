import os
import cv2
import random
import threading
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import easyocr
import pyttsx3
from transformers import pipeline
from googletrans import Translator
from Prompts import LLM
from dotenv import load_dotenv
import numpy as np
from Barcode import BarcodeProcessor
import io
import time
from client_server_connector import send_image_and_text
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr
from googletrans import Translator

load_dotenv()
llm = LLM()
barcode = BarcodeProcessor()
history_questions = ["", "", "", "", ""]
history_answers = ["", "", "", "", ""]
pipe = pipeline("image-segmentation", model="badmatr11x/semantic-image-segmentation")
print("OCR model loaded")
reader = easyocr.Reader(["en"], gpu=True)

recognizer = sr.Recognizer()
translator = Translator()

language_map = {
    'hi': 'hi',  
    'en': 'en',  
    'es': 'es',  
    'kn':'kn'
}


def recognize_text(image):
    text_detections = reader.readtext(image)
    recognized_texts = []
    if text_detections:
        for bbox, text, score in text_detections:
            if score > 0.25:
                recognized_texts.append(text)
    else:
        print("No text detected")
    return ", ".join(recognized_texts) if recognized_texts else "No text detected"


def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def get_largest_contour(binary_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(
        cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return max(contours, key=cv2.contourArea) if contours else None


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
        binary_mask = np.where(mask > 0, 1, 0).astype(np.uint8)
        largest_contour = get_largest_contour(binary_mask)
        if (
            largest_contour is not None
            and cv2.contourArea(largest_contour) >= min_area_threshold
        ):
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
    index = llm.get_object_of_interest(output_string, prompt).lower()
    if index in labels:
        i = labels.index(index)
        (x, y, w, h) = bounding_boxes[i]
        imghat = imghat[y : y + h, x : x + w]
        final_ocr = recognize_text(imghat)
        print("ITS WRITTEN :", final_ocr)
        return final_ocr
    return "Object not found."


def function_model(image, text):
    ret = send_image_and_text(image, text)
    ret = ret.split("Assistant")
    ret = ret[1]
    ret = ret[2 : len(ret) - 2]
    return ret


def bar_code(frame, text):
    reply = ""
    barcode.decode(frame)
    barcode.lookup()
    reply = barcode.barcode_llm(text)
    return reply


def decision_gen(frame, text, lbl_output, history_questions, history_answers):
    if not text:
        text = "What do you see?"
        prompt = text
    prompt = text
    val = str(llm.decision(text, history_questions, history_answers))
    print(val)
    ques = text
    if "0" in val:
        print("OCR")
        image = cv2.imread(r"captured_image.jpg")
        val2 = ocr_funct(prompt, image)
        val3 = llm.generate_for_ocr(val2, prompt)
        history_questions.pop(0)
        history_answers.pop(0)
        history_questions.append(ques)
        history_answers.append(val3)
    elif "1" in val:
        print("VQA")
        pil_image = Image.fromarray(frame)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0)
        frame = Image.open(buffer)
        val2 = function_model(frame, text)
        history_questions.pop(0)
        history_answers.pop(0)
        history_questions.append(ques)
        history_answers.append(val2)
    elif "2" in val:
        pass
    elif "3" in val:
        print("Barcode")
        val2 = bar_code(frame, text)
        history_questions.pop(0)
        history_answers.pop(0)
        history_questions.append(ques)
        history_answers.append(val2)
    else:
        print("History")
        history_questions.pop(0)
        history_answers.pop(0)
        history_questions.append(ques)
        history_answers.append(val)
    return str(history_answers[-1])


def speech_to_text(lbl_output):
    try:
        with sr.Microphone() as source:
            lbl_output.config(text="Speak something...")
            audio = recognizer.listen(source)

        text = recognizer.recognize_google(audio)
        lbl_output.config(text=f"Recognized speech: {text}")
        print(f"Recognized speech: {text}")

        detected_language = translator.detect(text).lang
        print(f"Detected language: {detected_language}")

        if detected_language == 'en':
            lbl_output.config(text="Detected English, no translation needed.")
            return text, detected_language
        else:
            simple_language_code = language_map.get(detected_language, detected_language)
            translated_text = translator.translate(text, dest=simple_language_code)
            print(f"Text in original script: {translated_text.text}")
            return translated_text.text, detected_language
    except sr.UnknownValueError:
        lbl_output.config(text="Sorry, could not understand the audio.")
    except sr.RequestError as e:
        lbl_output.config(text=f"Request error: {e}")
    except Exception as e:
        lbl_output.config(text=f"An error occurred: {e}")
    return "", ""


def text_to_speech(text, language_code):
    try:
        tts = gTTS(text=text, lang=language_code, slow=False)
        filename = "output.mp3"
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    except Exception as e:
        print(f"An error occurred: {e}")



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
    capture_thread = threading.Thread(target=capture_image_thread)
    capture_thread.start()


def capture_image_thread():
    global lbl_output, cap
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("captured_image.jpg", frame)
        lbl_output.config(text="Image captured!")
        text, detected_language = speech_to_text(lbl_output)

        if text:
            lbl_output.config(text=f"You said: {text}")
            text_to_speech(f"You said: {text}", 'en')

            reply = decision_gen(frame, text, lbl_output, history_questions, history_answers)

            translated_text = translator.translate(reply, dest='en')
            lbl_output.config(text=f"Assistant: {translated_text.text}")

            again_translate_text = translator.translate(translated_text.text, dest=detected_language)
            text_to_speech(again_translate_text.text, detected_language)


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
if not cap.isOpened():
    print("Error: Could not open video.")
    quit()

root = tk.Tk()
root.title("Pixel2Perception")
root.geometry("800x800")

frame_title = tk.Frame(root, bg=background_color)
frame_title.pack(pady=(20, 10), fill=tk.X)

img = Image.open("blind.png").resize((50, 50), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)

lbl_image = Label(frame_title, image=img, bg=background_color)
lbl_image.pack(side=tk.LEFT)

title_label = Label(
    frame_title,
    text="Pixel2Perception",
    bg=background_color,
    fg=text_color,
    font=("Helvetica", 20),
)
title_label.pack(side=tk.LEFT, padx=10)

frame_video = tk.Frame(root)
frame_video.pack()

lbl = Label(frame_video)
lbl.pack()

frame_buttons = tk.Frame(root, bg=background_color)
frame_buttons.pack(pady=20)

capture_button = Button(
    frame_buttons, text="Capture", command=capture_image, **button_style
)
capture_button.pack(side=tk.LEFT, padx=(0, 10))

quit_button = Button(frame_buttons, text="Quit", command=quit_app, **button_style)
quit_button.pack(side=tk.LEFT)

lbl_output = Label(
    root, text="", bg=background_color, fg=text_color, font=("Helvetica", 12)
)
lbl_output.pack(pady=(20, 0))

update_frame()
root.mainloop()
