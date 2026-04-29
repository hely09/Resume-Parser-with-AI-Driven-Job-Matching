"""
Course Recommender - Recommends courses for missing skills
"""
import re
import pandas as pd
from src.config import TOP_N_COURSES_PER_SKILL, MAX_SKILLS_FOR_COURSES


class CourseRecommender:
    """Recommend courses for missing skills"""

    def __init__(self, courses_csv_path=None):
        self.courses_df = None
        self.skill_to_courses = {}

        if courses_csv_path:
            self.load_courses_dataset(courses_csv_path)

    def load_courses_dataset(self, csv_path):
        """Load courses dataset"""
        print(f"→ Loading courses dataset from: {csv_path}")
        self.courses_df = pd.read_csv(csv_path)

        required_cols = ['name', 'institution', 'course_url']
        missing_cols = [col for col in required_cols if col not in self.courses_df.columns]

        if missing_cols:
            print(f"⚠ Warning: Missing columns {missing_cols}")
            return

        print(f"✓ Loaded {len(self.courses_df)} courses")
        self._build_skill_course_mapping()

    def _build_skill_course_mapping(self):
        """Build mapping from skills to courses"""
        print("→ Building skill-to-course mapping...")

        for _, row in self.courses_df.iterrows():
            course_name = str(row['name']).lower()
            course_info = {
                'course_name': row['name'],
                'platform': row['institution'],
                'url': row['course_url']
            }

            # Extract keywords
            keywords = re.findall(r'\b\w+\b', course_name)

            for keyword in keywords:
                if len(keyword) > 3:
                    if keyword not in self.skill_to_courses:
                        self.skill_to_courses[keyword] = []
                    if course_info not in self.skill_to_courses[keyword]:
                        self.skill_to_courses[keyword].append(course_info)

        print(f"✓ Mapped {len(self.skill_to_courses)} keywords to courses")

    def get_courses_for_skill(self, skill, top_n=TOP_N_COURSES_PER_SKILL):
        """Get courses for a specific skill"""
        skill_lower = skill.lower()

        # Exact match
        if skill_lower in self.skill_to_courses:
            return self.skill_to_courses[skill_lower][:top_n]

        # Partial matches
        matching_courses = []
        for keyword, courses in self.skill_to_courses.items():
            if skill_lower in keyword or keyword in skill_lower:
                matching_courses.extend(courses)

        if matching_courses:
            unique_courses = []
            seen = set()
            for course in matching_courses:
                course_id = course['course_name']
                if course_id not in seen:
                    seen.add(course_id)
                    unique_courses.append(course)
            return unique_courses[:top_n]

        # Fallback
        return [
            {
                'course_name': f'Search "{skill}" on Udemy',
                'platform': 'Udemy',
                'url': f'https://www.udemy.com/courses/search/?q={skill.replace(" ", "+")}'
            },
            {
                'course_name': f'Search "{skill}" on Coursera',
                'platform': 'Coursera',
                'url': f'https://www.coursera.org/search?query={skill.replace(" ", "+")}'
            }
        ]

    def recommend_courses_for_missing_skills(self, missing_skills,
                                            top_n_per_skill=TOP_N_COURSES_PER_SKILL,
                                            max_skills=MAX_SKILLS_FOR_COURSES):
        """Recommend courses for missing skills"""
        recommendations = []

        for skill in missing_skills[:max_skills]:
            courses = self.get_courses_for_skill(skill, top_n=top_n_per_skill)
            recommendations.append({
                'skill': skill,
                'courses': courses,
                'num_courses': len(courses)
            })

        return recommendations