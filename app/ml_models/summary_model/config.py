import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL_NAME = 'gemini-1.0-pro'

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

DEFAULT_MODEL_GENERATION_CONFIG = {
    'temperature': 0.9,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 10000,
}

DEFAULT_MODEL_SAFETY_CONFIG = [
    {
        'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
    },
    {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
    },
    {
        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
    },
    {
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_MEDIUM_AND_ABOVE'
    },
]

DEFAULT_SYSTEM_PROMPT = """
    You are a professional English summarizer working for a video call analysis company. Your job is to summarize video call conversations, extracting the most important bits of the conversation and compiling them into a summary of about 200-300 words. You can freely use the speaker's names as well. You will receive the transcript of the conversation in the form of JSON object with the following schema: 
            {
                'conversation': [
                    {
                        'speaker': 'A',
                        'message': 'Hello, how are you?',
                    },
                    {
                        'speaker': 'B',
                        'message': 'I am fine, thank you.'
                    }
                ]
            }
        
        Please make sure your response is in the form of a JSON object only with the following schema: 
            {
                summary: 'Summary of the conversation.',
            }
"""

DEFAULT_MODEL_CHAT_HISTORY = [
    {
        'role': 'user',
        'parts': DEFAULT_SYSTEM_PROMPT
    },
    {
        'role': 'model',
        'parts': 'Okay, I will adhere to the given schema and generate video call conversation summaries for you, only in JSON format.'
    }
]
