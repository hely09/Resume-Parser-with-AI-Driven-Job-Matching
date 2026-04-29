# 📄 AI Resume Analyzer with Job Matching

An AI-powered **Resume Analysis & Job Matching System** that extracts skills from resumes, predicts job categories, compares with job descriptions, and provides personalized course recommendations.

---

## 🚀 Project Overview

This system uses **Natural Language Processing (NLP)** and **Machine Learning** to analyze resumes and match them with suitable job roles.

👉 As described in the project report :

* Extracts structured data from resumes
* Identifies technical skills
* Predicts job category using ML models
* Matches resume with job description using semantic similarity
* Suggests missing skills and learning resources

---

## ✨ Features

* 📄 Resume Parsing (PDF)
* 🧠 Skill Extraction using NLP (spaCy)
* 🎯 Job Category Prediction (ML Model)
* 📊 Skill Match Analysis
* 🤖 Semantic Matching (Sentence-BERT)
* ❌ Missing Skills Detection
* 📚 Course Recommendations (Coursera, Udemy)
* 📋 Job Description Matching (Optional)
* 📥 Download Analysis Report (JSON)

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **NLP:** spaCy, Regex
* **ML Models:** Scikit-learn, Gradient Boosting
* **Semantic Matching:** Sentence Transformers (BERT)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly

---

## 📂 Project Structure

```bash id="strc91"
AI_Lab_Project/
│
├── app_1.py                 # Main Streamlit app
├── requirements.txt
│
├── src/                     # Core logic
│   ├── analyzer.py
│   ├── resume_parser.py
│   ├── skill_matcher.py
│   ├── job_description_matcher.py
│   ├── course_recommender.py
│   ├── category_mapper.py
│   ├── model_trainer.py
│   └── config.py
│
├── models/                  # Trained ML models
│   ├── trained_model.pkl
│   ├── vectorizer.pkl
│   ├── label_encoder.pkl
│   └── category_skills_map.pkl
│
├── data/                    # Datasets
│   ├── synthetic_it_skills_dataset.csv
│   ├── Coursera_courses.csv
│   └── vector_db/
│
├── uploads/                 # Uploaded resumes
│
├── train_model.py           # Model training script
├── test_resume_parsing.py
```

---

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash id="cmd91"
git clone https://github.com/your-username/ai-resume-analyzer.git
cd ai-resume-analyzer
```

---

### 2. Create Virtual Environment (Recommended)

```bash id="cmd92"
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash id="cmd93"
pip install -r requirements.txt
```

---

### 4. Download spaCy Model

```bash id="cmd94"
python -m spacy download en_core_web_sm
```

---

### 5. Train Model (IMPORTANT)

```bash id="cmd95"
python train_model.py
```

👉 Required before running app (models folder must exist)

---

### 6. Run Application

```bash id="cmd96"
streamlit run app_1.py
```

---

## 📊 How It Works

According to your system architecture (page 3) :

1. 📥 Upload Resume (PDF)
2. 🧹 Data Preprocessing (clean text)
3. 🧠 Resume Parsing (NLP extraction)
4. 🎯 Category Prediction (ML model)
5. 🔍 Skill Matching (TF-IDF + Semantic similarity)
6. ❌ Missing Skill Detection
7. 📚 Course Recommendation
8. 📊 Dashboard Output (Streamlit UI)

---

## 📈 Outputs

Your system provides:

* 📋 Basic Info (Name, Email, Education)
* 🛠️ Extracted Skills
* 🎯 Predicted Job Category
* 📊 Skill Coverage %
* ❌ Missing Skills
* 📚 Recommended Courses
* 📄 Job Description Match Score

👉 As shown in report screenshots:

* Skill extraction 
* Category prediction 
* JD match analysis 
* Missing skills 

---

## ⚠️ Limitations

* PDF parsing may fail for scanned resumes
* Model accuracy depends on dataset quality
* Short skills (e.g., C, R) may be ambiguous
* Semantic matching is not 100% accurate

---

## 🔮 Future Improvements

* OCR support for scanned PDFs
* Real-time job API integration
* Advanced deep learning models
* Multi-language support
* Better semantic matching
