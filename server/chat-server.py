"""
Simple Flask server for handling AI Medical Assistant chat requests
This server interfaces with OpenAI's API to provide medical information

Setup:
1. pip install flask flask-cors openai python-dotenv
2. Create a .env file with: OPENAI_API_KEY=your_key_here
3. Run: python chat-server.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# OpenAI Configuration
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# System prompt for the medical assistant
SYSTEM_PROMPT = """You are an AI Medical Assistant designed to provide general health information and guidance.

IMPORTANT GUIDELINES:
- Provide helpful, accurate medical information based on current medical knowledge
- Always remind users that this is general information and not a substitute for professional medical advice
- For serious symptoms or emergencies, always advise users to seek immediate medical attention
- Never provide specific diagnoses - only discuss possibilities and recommend seeing a healthcare provider
- Be empathetic and supportive
- Ask clarifying questions when needed to provide better guidance
- Remind users about medication interactions and the importance of consulting their doctor
- If asked about medications, provide general information but always recommend consulting a pharmacist or doctor
- For mental health concerns, be especially sensitive and recommend professional help when appropriate

EMERGENCY SITUATIONS:
If a user describes symptoms of a medical emergency (chest pain, difficulty breathing, severe bleeding, stroke symptoms, etc.), immediately advise them to call 911 or go to the emergency room.

Your goal is to be helpful while being responsible and ensuring users get proper medical care when needed."""


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests and communicate with OpenAI"""
    try:
        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400

        message = data['message']
        history = data.get('history', [])

        # Build messages array for OpenAI
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            *history,
            {'role': 'user', 'content': message}
        ]

        # Call OpenAI API
        completion = client.chat.completions.create(
            model='gpt-4',  # or 'gpt-3.5-turbo' for faster/cheaper responses
            messages=messages,
            max_tokens=800,
            temperature=0.7,
            presence_penalty=0.1,
            frequency_penalty=0.1
        )

        ai_response = completion.choices[0].message.content

        # Return response
        return jsonify({
            'response': ai_response,
            'usage': {
                'prompt_tokens': completion.usage.prompt_tokens,
                'completion_tokens': completion.usage.completion_tokens,
                'total_tokens': completion.usage.total_tokens
            }
        })

    except Exception as e:
        print(f'OpenAI API Error: {e}')

        if hasattr(e, 'status_code'):
            if e.status_code == 401:
                return jsonify({'error': 'Invalid OpenAI API key. Please check your configuration.'}), 401
            elif e.status_code == 429:
                return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429

        return jsonify({'error': 'An error occurred processing your request. Please try again.'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    from datetime import datetime
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.utcnow().isoformat()
    })


# Serve static files from webapp directory
@app.route('/pages/<path:filename>')
def serve_page(filename):
    return send_from_directory('../webapp/pages', filename)


@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory('../webapp/css', filename)


@app.route('/')
def index():
    return send_from_directory('../webapp/pages', 'chat.html')


if __name__ == '__main__':
    port = int(os.getenv('PORT', 3000))

    print(f'🏥 AI Medical Assistant server running on port {port}')
    print(f'📡 API endpoint: http://localhost:{port}/api/chat')
    print(f'🌐 Frontend: http://localhost:{port}/pages/chat.html')

    if not os.getenv('OPENAI_API_KEY'):
        print('⚠️  WARNING: OPENAI_API_KEY not found in environment variables!')
        print('   Please create a .env file with your OpenAI API key.')

    app.run(debug=True, host='0.0.0.0', port=port)
