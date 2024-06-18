from flask import Flask, request, jsonify
import os
from openai import OpenAI

app = Flask(__name__)

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            model="gpt-3.5-turbo"
        )
        message = response.choices[0].message['content'].strip()
        return jsonify({'response': message})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Virtual Assistant Service!"

@app.route('/routes', methods=['GET'])
def list_routes():
    import urllib
    output = []
    for rule in app.url_map.iter_rules():
        methods = ','.join(rule.methods)
        url = urllib.parse.unquote(f"{rule}")
        line = f"{url:50s} {methods}"
        output.append(line)
    return "<pre>" + "\n".join(sorted(output)) + "</pre>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
