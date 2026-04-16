from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# 📂 Load documents
def load_documents():
    docs = []
    filenames = []
    for file in os.listdir("docs"):
        with open(os.path.join("docs", file), "r", encoding="utf-8") as f:
            docs.append(f.read())
            filenames.append(file)
    return docs, filenames

# 🌍 Detect Language (Simple logic)
def detect_language(text):
    if any('\u0900' <= ch <= '\u097F' for ch in text):
        if "आहे" in text or "आणि" in text:
            return "Marathi 🇮🇳"
        else:
            return "Hindi 🇮🇳"
    else:
        return "English 🌍"

# 🔄 Query Translation (basic dictionary)
def translate_query(query):
    translations = {
        "renewable energy": ["नवीकरणीय ऊर्जा", "नवीकरणीय ऊर्जा"],
        "solar energy": ["सौर ऊर्जा", "सौर ऊर्जा"],
    }
    
    query_lower = query.lower()
    if query_lower in translations:
        return [query] + translations[query_lower]
    return [query]

# 🚀 Main Route
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    
    if request.method == "POST":
        query = request.form["query"]
        
        docs, filenames = load_documents()
        multilingual_query = " ".join(translate_query(query))
        
        all_text = docs + [multilingual_query]
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_text)
        
        similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        
        ranked = sorted(zip(filenames, similarity, docs), key=lambda x: x[1], reverse=True)
        
        for file, score, content in ranked:
            lang = detect_language(content)
            results.append((file, score, lang))

    return render_template("index.html", results=results)

# ▶ Run app
if __name__ == "__main__":
    app.run(debug=True)
