import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import Word
from nltk.corpus import stopwords
import streamlit as st
import plotly.express as px

# ======================
# NLTK setup (downloads stopwords + wordnet once)
# ======================
nltk.download("stopwords")
nltk.download("wordnet")
stop_words = set(stopwords.words("english"))

# ======================
# User Input Cleaning Function
# ======================
def clean_user_input(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # remove special chars
    words = text.split()
    words = [w for w in words if w not in stop_words]  # remove stopwords
    words = [Word(w).lemmatize() for w in words]  # lemmatization
    return " ".join(words)

# ======================
# Skill Aliases
# ======================
skill_aliases = {
    "ml": ["machine learning"],
    "ai": ["artificial intelligence"],
    "js": ["javascript"],
    "py": ["python"],
    "sql": ["structured query language"],
    "dl": ["deep learning"],
    "nlp": ["natural language processing"],
    "html5": ["html"],
    "css3": ["css"],
    "c++": ["cpp", "c plus plus"],
    "c#": ["csharp"]
}

def expand_skills(user_skills):
    expanded = set()
    for skill in user_skills:
        expanded.add(skill)
        if skill in skill_aliases:
            expanded.update(skill_aliases[skill])
    return expanded

# ======================
# Load dataset
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("mydata/internships.csv")  # ✅ keep your file in repo/data/
    
    # Extract City column
    city_cols = [col for col in df.columns if col.startswith("city_")]
    def get_city(row):
        for col in city_cols:
            if row.get(col) == 1:
                if col == "city_work_from_home":
                    return "Work from home"
                return col.replace("city_", "").title()
        return "Not specified"
    df["City"] = df.apply(get_city, axis=1)
    
    return df

df = load_data()

# ======================
# TF-IDF Vectorizer
# ======================
@st.cache_resource
def load_vectorizer(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = load_vectorizer(df)

# ======================
# Recommendation Function
# ======================
def recommend_internships(job_title, skills, city, expected_stipend, top_n=5):
    job_title_clean = clean_user_input(job_title)
    skills_clean = clean_user_input(skills)
    
    user_query = job_title_clean + " " + skills_clean
    user_vector = vectorizer.transform([user_query])
    
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    df["similarity"] = similarity_scores
    
    filtered_df = df[df["Stipend_numeric"] >= expected_stipend].copy()
    
    user_skills_raw = [s.strip().lower() for s in skills.split(",") if s.strip()]
    user_skills = expand_skills(user_skills_raw)
    
    filtered_df["Matched skills"] = filtered_df["combined_text"].apply(
        lambda text: sum(1 for s in user_skills if s in text.lower())
    )
    filtered_df["Matched Skills List"] = filtered_df["combined_text"].apply(
        lambda text: ", ".join([s for s in user_skills if s in text.lower()])
    )
    
    filtered_df["city_match"] = filtered_df["City"].str.lower() == city.lower()
    
    filtered_df["priority_score"] = (
        filtered_df["similarity"] * 0.6 +
        filtered_df["Matched skills"] * 0.3 +
        filtered_df["city_match"].astype(int) * 0.1
    )
    
    filtered_df["Internship"] = filtered_df["Internship"].apply(
        lambda x: x.title() if isinstance(x, str) else x
    )
    
    recommendations = (
        filtered_df
        .sort_values(by=["priority_score"], ascending=False)
        .head(top_n)
    )
    
    recommendations = recommendations.rename(columns={
        "Company": "Company",
        "Internship": "Internship",
        "City": "City",
        "Stipend_numeric": "Stipend per week",
        "duration_weeks": "Internship duration (weeks)",
        "Matched skills": "Matched skills",
        "Matched Skills List": "Matched Skills List"
    })
    
    return recommendations[[
        "Company",
        "Internship",
        "City",
        "Stipend per week",
        "Internship duration (weeks)",
        "Matched skills",
        "Matched Skills List"
    ]]

# ======================
# Streamlit UI
# ======================
st.title("🎯 Internship Recommendation System")

job_title = st.text_input("Enter your desired job title:")
skills = st.text_area("Enter your skills (comma separated):")
city = st.text_input("Enter your preferred city (or 'work from home'):")
expected_stipend = st.number_input("Enter your expected weekly stipend:", min_value=0, step=500)

if st.button("Get Recommendations"):
    results = recommend_internships(job_title, skills, city, expected_stipend, top_n=5)
    
    if results.empty:
        st.warning("⚠️ No internships match your filters. Try lowering stipend or changing city.")
    else:
        st.write("### Top 5 Recommended Internships")
        st.dataframe(results, use_container_width=True)
        
        # Bar chart of skills matched
        fig = px.bar(
            results,
            x="Internship",
            y="Matched skills",
            color="Company",
            title="Skills Matched per Internship",
            text="Matched skills",
            labels={"Matched skills": "Number of Matched Skills"}
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        
        # Download as CSV
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv,
            file_name="recommended_internships.csv",
            mime="text/csv"
        )
