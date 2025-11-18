from flask import Flask, send_from_directory
import os

# Set up the app
app = Flask(__name__)

# Get the directory that this 'app.py' file is in
# This is where your 'index.html' file should also be
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    """Serves the index.html file."""
    # This tells Flask to send the 'index.html' file 
    # from the same directory this script is in.
    return send_from_directory(APP_ROOT, 'index.html')

if __name__ == '__main__':
    # '0.0.0.0' means it will be accessible from your phone's browser
    app.run(host='0.0.0.0', port=5000, debug=False)
