import os
import cv2
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class LLM:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def generate_for_ocr(self, input_text, prompt):
        system_message = """
        Act like an expert in assistive technology, specializing in providing descriptions for visually impaired individuals. 
        Your task is to ensure that every piece of text is transformed into a comprehensive, easy-to-understand narrative, 
        focusing on the main ideas while ignoring irrelevant characters or symbols.
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
            messages=[{"role": "user", "content": message_content}],
            model="llama-3.1-70b-versatile",
        )
        response_text = chat_completion.choices[0].message.content
        print(response_text)
        return response_text

    def decision(self, text, history_questions, history_answers):
        prompt = text
        message_content = f"""
            You are the interface for a vision-language model. Based on the user's query, you must decide the correct course of action from the following options:

            1. **Say "0"** if the question requires OCR (Optical Character Recognition), which is used to read specific text from an image.
            2. **Say "1"** if the question requires the VQA (Visual Question Answering) model to analyze and describe general visual information in an image.
            3. If the user's query can be answered using previously asked questions and answers, respond with the answer directly. 
            - You have access to two arrays: one containing the last 5 questions and the other containing the last 5 answers.
            - If the user explicitly asks to use previous information or if the query matches past interactions, respond directly with the relevant answer without saying "2."

            **Guidelines to follow:**
            - **Use "0"** when the query involves reading or extracting specific text from an image (OCR).
            - **Use "1"** when the query asks about general visual content or image-based analysis (VQA).
            - For matching questions in the previous data, return the stored answer directly.

            Previous interactions:
            - Last 5 questions: {history_questions}
            - Last 5 answers: {history_answers}

            User query: "{prompt}"

            Dont give explanation, just give an answer like an actual person.
            """

        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": message_content}],
            model="llama-3.1-70b-versatile",
        )

        val = chat_completion.choices[0].message.content
        print("DECISION : ", val)
        return val
