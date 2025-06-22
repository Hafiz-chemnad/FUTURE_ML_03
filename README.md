# AI-Powered Customer Service Chatbot
This project implements an AI-powered customer service chatbot using a neural network trained on a dataset of customer service interactions. The chatbot is designed to understand user queries and provide relevant responses or direct users to appropriate services (like contacting a human agent).

The project consists of two main parts:

**Model Training & Export**: A Google Colab notebook for data preprocessing, model training, and exporting the necessary model artifacts.

**FastAPI Backend & Web Interface**: A Python FastAPI application that serves as the backend for the chatbot, loads the trained model, makes predictions, and provides a simple web interface for user interaction.

## Project Structure
chatbot_project/
├── main.py                     # FastAPI application for the chatbot backend
├── chatbot_model.h5            # Trained Keras neural network model
├── words.pkl                   # Vocabulary used for bag-of-words
├── classes.pkl                 # List of intent tags (categories)
├── intents.json                # Chatbot's knowledge base (intents, patterns, responses)
├── requirements.txt            # Python dependencies for the FastAPI app
├── .gitignore                  # Specifies files/folders to ignore in Git
└── static/
    ├── index.html              # HTML structure for the chatbot interface
    ├── style.css               # CSS for styling the chatbot interface
    └── script.js               # JavaScript for frontend interaction with FastAPI

## Setup and Installation
Follow these steps to set up and run the chatbot on your local machine.

### Prerequisites
Python 3.8+ installed

pip (Python package installer)

git installed (for cloning this repository, if you weren't building from scratch)

### 1. Clone the Repository (If starting fresh from GitHub)
If you're setting this up on a new machine from GitHub, first clone the repository:

git clone https://github.com/Hafiz-chemnad/FUTURE_ML_03.git
cd FUTURE_ML_03 # Or whatever your repository name is

If you followed the previous instructions and are adding README.md to your existing local project, you can skip this step and ensure you are in your chatbot_project directory.

### 2. Set up Python Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies:

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

### 3. Install Dependencies
Install all the required Python libraries using the requirements.txt file:

pip install -r requirements.txt

**Note: Ensure the TensorFlow version in requirements.txt matches the version used during model training in Google Colab to avoid compatibility issues.**

### 4. NLTK Data Download
The main.py script attempts to download necessary NLTK data (punkt, wordnet, averaged_perceptron_tagger) automatically when it starts. If you face any issues, you can manually download them:

python -m nltk.downloader punkt
python -m nltk.downloader wordnet
python -m nltk.downloader averaged_perceptron_tagger

### 5. Model and Data Files
Ensure the following files are present in your project's root directory:

chatbot_model.h5

words.pkl

classes.pkl

intents.json

These files are generated and exported from the Google Colab notebook (Part 1 of the project setup). If you are cloning this repository, they should already be included.

### 6. Run the FastAPI Application
Once all dependencies are installed and files are in place, start the FastAPI server:

uvicorn main:app --reload

You should see output indicating that the server is running, typically on http://127.0.0.1:8000.

### 7. Access the Chatbot
#### Open your web browser and navigate to:

http://127.0.0.1:8000

You will see the chatbot's web interface, and you can start interacting with it!

#### How it Works
**Dataset**: The chatbot is trained on a custom dataset (Bitext_Sample_Customer_Service_Testing_Dataset.csv) containing utterance (user queries) and intent (predefined categories).

**intents.json**: This file defines the chatbot's knowledge base, mapping intent tags to patterns (example phrases) and responses. Crucially, the responses for each intent should be manually populated with helpful text relevant to that intent.

**NLTK**: Used for natural language processing tasks like tokenization and lemmatization.

**Bag-of-Words (BoW)**: User input is converted into a numerical Bag-of-Words representation.

**TensorFlow/Keras**: A neural network model (defined in the Colab notebook) is trained to classify the BoW input into one of the predefined intents.

**FastAPI**: Provides a lightweight and efficient web server to host the chatbot, handling API requests from the frontend and returning bot responses.

**HTML, CSS, JavaScript**: A simple web interface allows users to type messages and receive responses from the chatbot.

## Customization

**Update intents.json**: To expand the chatbot's capabilities or refine its responses, modify intents.json. Add new intents, patterns, or more diverse responses for existing ones. After modifying intents.json, you will need to re-run the Colab notebook (especially cells 2 and 3) to retrain the model and export the updated files.

**Train with More Data**: Provide a larger and more diverse dataset to improve the chatbot's accuracy and understanding.

**Adjust ERROR_THRESHOLD**: In main.py, the ERROR_THRESHOLD variable controls the confidence level required for an intent to be matched. Adjusting this can make the chatbot more or less likely to respond with a specific intent versus a "no_match" fallback.
