This project uses Natural Language Processing (NLP) and Machine Learning to classify text as AI-generated or human-written.

🚀 Features
Text classification (AI vs Human)
TF-IDF based feature extraction
Multinomial Naive Bayes model
Simple Streamlit web app interface
🛠️ Tech Stack
Python
Scikit-learn
Pandas & NumPy
Streamlit
📂 Project Structure
AI-vs-Human-Text-Detection/
├── ai_vs_human.ipynb
├── app.py
├── ml_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
▶️ How to Run
Install dependencies:
pip install -r requirements.txt
Run the app:
streamlit run app.py
📊 Model Details
Feature Extraction: TF-IDF (Term Frequency–Inverse Document Frequency)
Algorithm: Multinomial Naive Bayes
Task: Binary text classification (AI vs Human)
📌 Use Cases
Content authenticity detection
Academic integrity verification
Spam / automated text detectio
