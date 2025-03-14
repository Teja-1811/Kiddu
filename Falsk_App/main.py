from flask import Flask
from app.routes import init_routes
import os

app = Flask(__name__)

# Configure Upload Folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize Routes
init_routes(app)

if __name__ == '__main__':
    app.run(debug=True)
