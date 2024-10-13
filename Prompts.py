import os
import cv2
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class LLM:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def generate_for_ocr(self, input_text, prompt):
        message_content = f"""
        Describe this text recognized from OCR for a blind person. 

        OCR Text:
        {input_text}

        Instructions for Task Completion:
        - Your output should be a detailed description of the provided text for a blind person to understand.
        - Focus on capturing the essence and key details of the text.
        - Use easy English.
        - Ignore random letters and symbols.
        - Tell only about the main points.
        Answer the following question using all the info given to you:
        {prompt}
        """

        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": message_content}],
            model="llama-3.1-70b-versatile",
        )

        response_text = chat_completion.choices[0].message.content
        print(response_text)
        return response_text

    def get_object_of_interest(self, data, prompt):
        message_content = f"""
        We are building a vision-language model. You are to act as the interface.
        We used a semantic segmentation model to get objects and areas.
        These will be given to you as data: {data}
        Find the object with the closest reference to the object inquired about by the user.
        Use data of both names and object areas to decide which is the most feasible object, which will usually
        be the object with the biggest area referenced by the user. Print only the object name as is, no more no less.
        User prompt: {prompt}
        """

        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": message_content}],
            model="llama-3.1-70b-versatile",
        )

        response_text = chat_completion.choices[0].message.content
        print(response_text)
        return response_text

    def decision(self, text , history_questions, history_answers):
        prompt = text
        message_content = f"""
        We are building a vision-language model. You will act as the interface for this model.
        A user will ask a question, and you must decide the appropriate action based on the question:
        - Say "0" if the prompt requires OCR (Optical Character Recognition).
        - Say "1" if the prompt requires the use of the VQA (Visual Question Answering) model.
        - If the question can be answered using existing knowledge, respond with the answer directly. This existing knowledge is available as two arrays of strings, each containing the last 5 questions and answers.
          Note that in many cases the user might explicitly ask to use previous info.

        Remember:
        - Use "0" for OCR when specific text needs to be read.
        - Use "1" for VQA to act as the user's eyes for general visual information.
        - Answer directly if the question matches previous knowledge. If the answer is found in the previous knowledge, respond directly without saying "2".

        Here is the previous information:
        - Previous questions: {history_questions}
        - Previous answers: {history_answers}

        The prompt is: {prompt}
        """

        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": message_content}],
            model="llama-3.1-70b-versatile",
        )

        val = chat_completion.choices[0].message.content
        return val


