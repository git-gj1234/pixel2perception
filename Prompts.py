import os
import cv2
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class LLM:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def generate_for_ocr(self, input_text, prompt):
        print("PREDICTED : ", input_text)
        system_message = """
        You are a model that assists visually impaired users by interpreting and describing text recognized through Optical Character Recognition (OCR). 
        Your goal is to extract the essential information from the text, ignoring any random characters or irrelevant symbols, 
        and present it in a clear and concise manner.
        """

        user_message = f"""
        You are tasked with describing OCR-recognized text for a blind person. This text may contain random symbols or irrelevant characters, 
        so your job is to focus only on the key information.

        OCR Text:
        "{input_text}"

        Instructions for Task Completion:
        1. Provide a detailed description of the essential content of this OCR text.
        2. Use clear, simple language to ensure comprehension by a blind person.
        3. Ignore any random symbols, stray characters, or irrelevant information.
        4. Highlight the most important points clearly and avoid unnecessary details.

        Additionally, answer the following question using the provided text:
        "{prompt}"

        Just answer me realteted to the aksed question, dont give other information.
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

    def get_object_of_interest(self, data, prompt):
        system_message = """
        You are a model that specializes in object recognition and semantic segmentation analysis. 
        You help users by analyzing identified objects and their respective areas, 
        and you prioritize objects with the largest area when multiple matches exist.
        """

        message_content = f"""
        You are the interface for a vision-language model designed to analyze objects from a semantic segmentation model.
        The model provides data on identified objects and their areas as follows:
        {data}

        Your tasks:
        1. Find the object that best matches the user's inquiry by analyzing both the names and the areas of the identified objects.
        2. Prioritize objects with the largest area when multiple objects match or are referenced by the user.
        3. Based on this analysis, return **only** the object name that is the most relevant to the user's inquiry—no additional information.

        User prompt: "{prompt}"

        Take a deep breath and work on this problem step-by-step.
        """

        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": message_content},
            ],
            model="llama-3.1-70b-versatile",
        )

        response_text = chat_completion.choices[0].message.content
        print(response_text)
        return response_text

    def decision(self, text, history_questions, history_answers):
        system_message = """
        Your objective is to identify the type of processing needed for each query, 
        selecting between OCR (Optical Character Recognition), general visual analysis 
        (VQA), object location, barcode scanning, or referencing previous user interactions. 
        You will choose the correct method based on these guidelines:

        Guidelines:
        - Say "0" if the query requires OCR, for example when the task involves reading or 
        extracting specific text from an image.
        - Say "1" if the query involves general image analysis using a VQA (Visual Question Answering)
          model to describe or interpret the content of an image.
        - Say "2" if the query involves locating an object within a room or environment, 
        such as identifying the position of specific items like a bottle, keys, or furniture.
        - Say "3" if the task involves scanning a barcode and retrieving related product details.
        - If the user query can be answered using previous interactions, you must respond 
        directly with the stored answer without saying "4." You have access to an array containing the last 5 questions and their corresponding answers:
            -Last 5 questions: {history_questions}
            -Last 5 answers: {history_answers}
        If the current question matches any of the previous ones or the user explicitly 
        refers to past interactions, retrieve and return the relevant answer immediately.
        
        Step-by-Step Decision Process:
        - Classify the user query based on the type of visual processing needed (OCR, VQA,
          object detection, barcode scanning, or history-based answer retrieval).
        - Select the appropriate response method (from 0 to 3) based on the guidelines above.
        - Respond quickly and accurately without providing unnecessary details, unless the 
        query can be matched with previous history, in which case provide the relevant answer from stored responses.
        
        Take a deep breath and work on this problem step-by-step.

        """

        prompt = text
        message_content = f"""
        You are the interface for a vision-language model. Based on the user's query, you must decide the correct course of action from the following options:

        User query: "{prompt}"

        Don't give explanation, just give an answer like an actual person.
        """

        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": message_content},
            ],
            model="llama-3.1-70b-versatile",
        )

        val = chat_completion.choices[0].message.content
        return val
