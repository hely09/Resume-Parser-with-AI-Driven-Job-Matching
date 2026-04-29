"""
Job Description Matcher - Match resume against job descriptions
"""
import re
from collections import Counter
from src.skill_matcher import SemanticSkillMatcher


class JobDescriptionMatcher:
    """Match resume against job descriptions"""

    def __init__(self, semantic_matcher=None):
        self.semantic_matcher = semantic_matcher or SemanticSkillMatcher()

        # Common job-related keywords to extract
        self.skill_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized terms
            r'\b(?:python|java|javascript|react|angular|vue|node|sql|mongodb|aws|azure|gcp|docker|kubernetes|git|agile|scrum)\b',
        ]

        # Role/position keywords
        self.role_keywords = {
            'developer': ['developer', 'engineer', 'programmer', 'coder'],
            'senior': ['senior', 'lead', 'principal', 'staff'],
            'junior': ['junior', 'entry', 'associate'],
            'manager': ['manager', 'director', 'head'],
            'designer': ['designer', 'ux', 'ui'],
            'analyst': ['analyst', 'data scientist', 'researcher'],
            'architect': ['architect', 'technical lead']
        }

    def extract_jd_requirements(self, jd_text):
        """Extract requirements from job description"""
        jd_lower = jd_text.lower()

        # Extract skills
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9+#.]{2,}\b', jd_text)
        skills = []

        # Filter to likely technical skills
        for word in words:
            word_clean = word.strip('.,;:')
            # Keep capitalized words or known tech terms
            if (word[0].isupper() and len(word_clean) > 2) or \
                    word_clean.lower() in ['python', 'java', 'sql', 'aws', 'git', 'react', 'node', 'docker']:
                skills.append(word_clean)

        # Get unique skills
        skill_counts = Counter(skills)
        important_skills = [skill for skill, count in skill_counts.most_common(50)]

        # Extract experience requirements
        experience_match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience', jd_lower)
        years_required = int(experience_match.group(1)) if experience_match else None

        # Extract education requirements
        education_keywords = ['bachelor', 'master', 'phd', 'degree', 'bs', 'ms', 'ba', 'ma']
        education_required = any(keyword in jd_lower for keyword in education_keywords)

        # Detect role level
        role_level = 'mid'
        if any(keyword in jd_lower for keyword in self.role_keywords['senior']):
            role_level = 'senior'
        elif any(keyword in jd_lower for keyword in self.role_keywords['junior']):
            role_level = 'junior'
        elif any(keyword in jd_lower for keyword in self.role_keywords['manager']):
            role_level = 'manager'

        return {
            'skills': important_skills,
            'years_required': years_required,
            'education_required': education_required,
            'role_level': role_level,
            'jd_text': jd_text
        }

    def match_resume_to_jd(self, resume_skills, jd_requirements, semantic_threshold=0.75):
        """Match resume skills against job description requirements"""
        jd_skills = jd_requirements['skills']

        # Exact matches
        resume_skills_lower = [skill.lower() for skill in resume_skills]
        jd_skills_lower = [skill.lower() for skill in jd_skills]

        exact_matches = []
        for jd_skill in jd_skills:
            if jd_skill.lower() in resume_skills_lower:
                exact_matches.append(jd_skill)

        # Semantic matches using the matcher
        semantic_matches = {}
        unmatched_jd_skills = [skill for skill in jd_skills if skill.lower() not in [m.lower() for m in exact_matches]]

        for jd_skill in unmatched_jd_skills:
            # Find similar resume skills
            jd_embedding = self.semantic_matcher._get_embedding(jd_skill)

            for resume_skill in resume_skills:
                if resume_skill.lower() in [m.lower() for m in exact_matches]:
                    continue

                resume_embedding = self.semantic_matcher._get_embedding(resume_skill)
                similarity = self.semantic_matcher._cosine_similarity(jd_embedding, resume_embedding)

                if similarity >= semantic_threshold:
                    if jd_skill not in semantic_matches:
                        semantic_matches[jd_skill] = []
                    semantic_matches[jd_skill].append({
                        'resume_skill': resume_skill,
                        'similarity': float(similarity)
                    })

        # Sort semantic matches by similarity
        for jd_skill in semantic_matches:
            semantic_matches[jd_skill] = sorted(
                semantic_matches[jd_skill],
                key=lambda x: x['similarity'],
                reverse=True
            )[:3]  # Top 3 matches

        # Missing skills
        all_matched = set([m.lower() for m in exact_matches] + list(semantic_matches.keys()))
        missing_skills = [skill for skill in jd_skills if skill.lower() not in all_matched]

        # Calculate match percentage
        total_jd_skills = len(jd_skills)
        matched_skills = len(exact_matches) + len(semantic_matches)
        match_percentage = (matched_skills / total_jd_skills * 100) if total_jd_skills > 0 else 0

        return {
            'match_percentage': match_percentage,
            'total_jd_requirements': total_jd_skills,
            'exact_matches': sorted(exact_matches),
            'exact_match_count': len(exact_matches),
            'semantic_matches': semantic_matches,
            'semantic_match_count': len(semantic_matches),
            'missing_skills': sorted(missing_skills),
            'missing_count': len(missing_skills),
            'role_level': jd_requirements['role_level'],
            'years_required': jd_requirements['years_required']
        }

    def calculate_role_fit(self, match_results, resume_info):
        """Calculate overall role fit score"""
        # Base score from skill matching
        skill_score = match_results['match_percentage']

        # Adjust for role level (you can enhance this with actual experience from resume)
        role_level = match_results['role_level']
        level_fit = 'good'  # Default

        if role_level == 'senior' and skill_score < 70:
            level_fit = 'below_expected'
        elif role_level == 'junior' and skill_score > 60:
            level_fit = 'exceeds_expected'

        # Overall fit categories
        if skill_score >= 80:
            fit_category = 'Excellent Fit'
            fit_description = 'Strong match with most requirements met'
        elif skill_score >= 60:
            fit_category = 'Good Fit'
            fit_description = 'Good match with some skill gaps'
        elif skill_score >= 40:
            fit_category = 'Moderate Fit'
            fit_description = 'Partial match, significant upskilling needed'
        else:
            fit_category = 'Low Fit'
            fit_description = 'Limited match, consider other opportunities'

        # Key strengths
        key_strengths = match_results['exact_matches'][:5]

        # Priority gaps
        priority_gaps = match_results['missing_skills'][:5]

        return {
            'overall_score': skill_score,
            'fit_category': fit_category,
            'fit_description': fit_description,
            'role_level': role_level,
            'level_fit': level_fit,
            'key_strengths': key_strengths,
            'priority_gaps': priority_gaps,
            'recommendation': self._get_recommendation(skill_score, role_level)
        }

    def _get_recommendation(self, score, role_level):
        """Get application recommendation"""
        if score >= 75:
            return 'Highly recommended to apply - strong profile match'
        elif score >= 60:
            return 'Recommended to apply - good match with minor gaps'
        elif score >= 45:
            return 'Consider applying after addressing key skill gaps'
        else:
            return 'Focus on upskilling before applying'

    def generate_jd_analysis(self, resume_skills, jd_text, resume_info=None, semantic_threshold=0.75):
        """Complete job description analysis"""
        # Extract JD requirements
        jd_requirements = self.extract_jd_requirements(jd_text)

        # Match resume to JD
        match_results = self.match_resume_to_jd(
            resume_skills,
            jd_requirements,
            semantic_threshold
        )

        # Calculate role fit
        role_fit = self.calculate_role_fit(match_results, resume_info or {})

        return {
            'jd_requirements': jd_requirements,
            'match_results': match_results,
            'role_fit': role_fit
        }