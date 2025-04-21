
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify

nlp = spacy.load("en_core_web_sm")

def mask_pii(email_text):
    doc = nlp(email_text)
    masked_text = email_text
    masked_entities = []
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            masked_text = masked_text.replace(ent.text, "[full_name]")
            masked_entities.append({"position": [ent.start_char, ent.end_char], "classification": "full_name", "entity": ent.text})
        elif ent.label_ == "EMAIL":
            masked_text = masked_text.replace(ent.text, "[email]")
            masked_entities.append({"position": [ent.start_char, ent.end_char], "classification": "email", "entity": ent.text})
    
    masked_text = re.sub(r"\d{3}[-.]?\d{3}[-.]?\d{4}", "[phone_number]", masked_text)
    for m in re.finditer(r"\d{3}[-.]?\d{3}[-.]?\d{4}", email_text):
        masked_entities.append({"position": [m.start(), m.end()], "classification": "phone_number", "entity": m.group()})
    
    return masked_text, masked_entities

def classify_email(email_text, vectorizer, clf):
    masked_text, masked_entities = mask_pii(email_text)
    tfidf = vectorizer.transform([masked_text])
    category = clf.predict(tfidf)[0]
    return masked_text, masked_entities, category

def train_model(masked_emails, categories):
    X_train, X_test, y_train, y_test = train_test_split(masked_emails, categories, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_tfidf, y_train)
    return vectorizer, clf

app = Flask(__name__)

masked_emails = []
categories = []
vectorizer, clf = train_model(masked_emails, categories)

@app.route("/classify", methods=["POST"])
def classify():
    email_text = request.json["email"]
    masked_text, masked_entities, category = classify_email(email_text, vectorizer, clf)
    response = {
        "input_email_body": email_text,
        "list_of_masked_entities": masked_entities,
        "masked_email": masked_text,
        "category_of_the_email": category
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
