Ai chatbot - This project is a simple AI chatbot built using Python, TensorFlow, and Flask. It can respond to basic conversational prompts, answer common questions from secondary school subjects like Math, English, and Science, and simulate natural human-like interactions.
Features
Intent classification using a neural network (Keras + TensorFlow)

Pattern-based response selection from a custom intents.json file

Real-time chat interface with Flask backend and HTML frontend

Supports:

Basic greetings and goodbyes

Simple arithmetic questions

English synonyms and antonyms

General science definitions and concepts

How It Works
User inputs a message

Model classifies the intent using a bag-of-words + softmax classifier

Response is selected based on intent match from predefined options

Tech Stack
Python

Flask (backend)

TensorFlow/Keras (intent classifier)

HTML/CSS/JavaScript (frontend)

NLTK for tokenization and lemmatization

Future Improvements
Dynamic response generation using large language models (LLMs)

Voice input and text-to-speech output

Integration with external knowledge sources (e.g., Wikipedia, WolframAlpha)
