# Chatbot Project

This project is a chatbot built using Python, TensorFlow, and Natural Language Processing (NLP). The chatbot is trained on predefined intents and can classify user input into different categories.

## Project Description
This chatbot project consists of three main files:
1. **chatbot_training.py** - Preprocesses data and trains a chatbot model using neural networks.
2. **chatbot_response.py** - Loads the trained model and processes user input to provide responses.
3. **intents.json** - Contains predefined intents, user queries, and corresponding chatbot responses.

The chatbot is designed to assist users with queries related to Britsure, an insurance tech company offering digital insurance solutions and job opportunities.

## Prerequisites
Ensure you have Python installed (recommended: Python 3.8+). You can download it from [python.org](https://www.python.org/downloads/).

## Setting Up the Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

### Create a Virtual Environment
```sh
python -m venv chatbot_env
```

### Activate the Virtual Environment
- **Windows (Command Prompt):**
  ```sh
  chatbot_env\Scripts\activate
  ```
- **Windows (PowerShell):**
  ```sh
  chatbot_env\Scripts\Activate.ps1
  ```
- **Mac/Linux:**
  ```sh
  source chatbot_env/bin/activate
  ```

## Install Dependencies
Once the virtual environment is activated, install the required libraries:
```sh
pip install tensorflow numpy nltk pickle5
```

### Download NLTK Data
Ensure you have the required NLTK data files:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Running the Chatbot Training Script
Run the script to preprocess data and train the chatbot model:
```sh
python chatbot_training.py
```

This script will:
- Load and process the intents JSON file
- Create a bag-of-words representation
- Train a neural network model using TensorFlow
- Save the trained model as `chatbot_model.h5`

## Running the Chatbot Response Script
To interact with the chatbot after training, run:
```sh
python chatbot_response.py
```
This script will:
- Load the trained model and word data
- Process user input to classify intent
- Respond based on predefined intents in `intents.json`

## Saving and Loading Model Data
The processed words and class labels are stored using `pickle`:
- `words.pkl`
- `classes.pkl`

The trained model is saved as:
- `chatbot_model.h5`

## Deactivating the Virtual Environment
Once you're done, deactivate the virtual environment:
```sh
deactivate
```

## Git Commands
### Initialize a Git Repository
```sh
git init
```

### Add and Commit Files
```sh
git add .
git commit -m "Initial commit"
```

### Push to GitHub (Replace with your repository URL)
```sh
git remote add origin https://github.com/yourusername/chatbot.git
git branch -M main
git push -u origin main
```

## License
This project is licensed under the MIT License.

