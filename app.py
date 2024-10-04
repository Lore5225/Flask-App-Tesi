from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from flask import Flask, request, jsonify
from flask_cors import CORS  
import logging
import re

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

def format_output(text):
    """Convert Markdown bold syntax to HTML strong tags."""
    return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

def initialise_llama3_2B():
    try:
        create_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a teacher creating exams for students."),
                ("user", "Generate an exam that includes various questions based on a specific subject like databases. The exam should include multiple-choice, true/false, and short-answer questions.")
            ]
        )

        llama_model = Ollama(model="llama3")
        output_parser = StrOutputParser()

        # Create the chain
        chatbot_pipeline = create_prompt | llama_model | output_parser
        return chatbot_pipeline
    except Exception as e:
        logging.error(f"Failed to initialize chatbot: {e}")
        raise

chatbot_pipeline = initialise_llama3_2B()

@app.route('/genera-esame', methods=['POST'])
def genera_esame():
    data = request.get_json()
    query_input = data.get('topic', 'general exam')  # Use 'general exam' as default topic if not provided
    output = None
    try:
        response = chatbot_pipeline.invoke({'question': f"Generate an exam for {query_input}."})
        output = format_output(response)
    except Exception as e:
        logging.error(f"Error during chatbot invocation: {e}")
        output = "Sorry, an error occurred while processing your request."
    return jsonify({'output': output})

if __name__ == '__main__':
    app.run(debug=True)
