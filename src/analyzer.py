"""
Resume Analyzer - Main analysis system
"""
import json
from datetime import datetime
from src.resume_parser import parse_resume
from src.model_trainer import load_trained_models
from src.skill_matcher import SemanticSkillMatcher, HybridSkillMatcher
from src.course_recommender import CourseRecommender
from src.config import SEMANTIC_THRESHOLD


class ResumeAnalyzer:
    """Complete resume analysis system"""

    def __init__(self, courses_csv_path=None):
        print("\n" + "=" * 70)
        print("INITIALIZING RESUME ANALYZER")
        print("=" * 70)

        # Load trained models
        print("\n→ Loading trained models...")
        model_data = load_trained_models()

        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.category_skills_map = model_data['category_skills_map']

        print("✓ Models loaded successfully")
        print(f"✓ Categories: {len(self.category_skills_map)}")

        # Initialize semantic matcher
        print("\n→ Initializing semantic matcher...")
        self.semantic_matcher = SemanticSkillMatcher()
        self.semantic_matcher.set_category_skills(self.category_skills_map)
        self.semantic_matcher.precompute_embeddings()

        # Initialize hybrid matcher
        self.hybrid_matcher = HybridSkillMatcher(
            self.semantic_matcher,
            self.category_skills_map
        )
        print("✓ Skill matchers initialized")

        # Initialize course recommender
        print("\n→ Initializing course recommender...")
        self.course_recommender = CourseRecommender(courses_csv_path)

        print("\n" + "=" * 70)
        print("✅ SYSTEM READY")
        print("=" * 70)

    def predict_category(self, extracted_skills, top_n=3):
        """Predict job category from skills"""
        if not extracted_skills:
            return []

        skills_text = ' '.join([skill.lower() for skill in extracted_skills])
        skills_vec = self.vectorizer.transform([skills_text])
        probabilities = self.model.predict_proba(skills_vec)[0]
        top_indices = probabilities.argsort()[-top_n:][::-1]

        results = []
        for idx in top_indices:
            category = self.label_encoder.classes_[idx]
            confidence = probabilities[idx]
            results.append({
                'category': category,
                'confidence': float(confidence),
                'confidence_percentage': f"{confidence * 100:.1f}%"
            })

        return results

    def analyze_resume(self, pdf_path, semantic_threshold=SEMANTIC_THRESHOLD, debug=False):
        """Complete resume analysis"""
        print("\n" + "=" * 70)
        print("ANALYZING RESUME")
        print("=" * 70)

        # Parse resume
        print("\n→ Parsing resume...")
        parsed_data = parse_resume(pdf_path, debug=debug)
        print(f"✓ Extracted {len(parsed_data['skills'])} skills")

        if not parsed_data['skills']:
            print("❌ No skills found!")
            return None

        # Predict category
        print("\n→ Predicting job category...")
        predicted_categories = self.predict_category(parsed_data['skills'], top_n=3)
        top_category = predicted_categories[0]['category'] if predicted_categories else None

        if not top_category:
            print("❌ Could not predict category!")
            return None

        print(f"✓ Top prediction: {top_category}")

        # Analyze skills
        print("\n→ Analyzing skill coverage...")
        coverage_report = self.hybrid_matcher.get_skill_coverage_report(
            parsed_data['skills'],
            top_category,
            semantic_threshold
        )
        print(f"✓ Coverage: {coverage_report['coverage_percentage']:.1f}%")

        # Get course recommendations
        print("\n→ Generating course recommendations...")
        missing_skills = coverage_report['missing_skills']
        course_recommendations = self.course_recommender.recommend_courses_for_missing_skills(
            missing_skills
        )
        print(f"✓ Generated recommendations for {len(course_recommendations)} skills")

        # Compile analysis
        analysis = {
            'metadata': {
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'resume_file': pdf_path
            },
            'basic_info': {
                'name': parsed_data['name'],
                'email': parsed_data['email'],
                'contact': parsed_data['contact'],
                'education': parsed_data['education']
            },
            'skills': {
                'extracted_skills': parsed_data['skills'],
                'total_skills': len(parsed_data['skills'])
            },
            'category_prediction': {
                'top_predictions': predicted_categories,
                'selected_category': top_category
            },
            'skill_analysis': coverage_report,
            'course_recommendations': course_recommendations
        }

        print("\n✅ Analysis complete!")
        return analysis

    def save_analysis(self, analysis, output_path):
        """Save analysis to JSON"""
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"✓ Analysis saved to {output_path}")