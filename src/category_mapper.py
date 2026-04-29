"""
Category Mapper - Maps skills to appropriate categories using your JSON data
"""
import json
from pathlib import Path


class CategoryMapper:
    """Maps skills to categories using your category-skills JSON data"""

    def __init__(self, category_skills_map=None):
        """
        Initialize with category-skills mapping

        Args:
            category_skills_map: Dict like {'Data Scientist': ['python', 'ml', ...], ...}
        """
        self.category_skills_map = category_skills_map or {}
        self.skill_to_category_map = {}

        if self.category_skills_map:
            self._build_reverse_mapping()

    def _build_reverse_mapping(self):
        """
        Build reverse mapping: skill -> category
        """
        for category, skills in self.category_skills_map.items():
            for skill in skills:
                skill_lower = skill.lower().strip()
                # If skill appears in multiple categories, keep first one
                # or you could store all categories per skill
                if skill_lower not in self.skill_to_category_map:
                    self.skill_to_category_map[skill_lower] = category

    def get_category_from_skills(self, skills_list):
        """
        Determine category based on skill keywords from your data
        Returns most likely category based on skill matches

        Args:
            skills_list: List of skills extracted from resume

        Returns:
            tuple: (category_name, confidence_score)
        """
        if not skills_list or not self.skill_to_category_map:
            return None, 0.0

        # Count category matches
        category_scores = {}
        total_matches = 0

        for skill in skills_list:
            skill_lower = skill.lower().strip()

            # Check if skill exists in your data
            if skill_lower in self.skill_to_category_map:
                category = self.skill_to_category_map[skill_lower]
                category_scores[category] = category_scores.get(category, 0) + 1
                total_matches += 1

        # Return category with highest score
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            category_name = best_category[0]
            match_count = best_category[1]

            # Calculate confidence based on how many skills matched
            confidence = min(match_count / len(skills_list), 0.95)

            return category_name, confidence

        return None, 0.0

    def refine_category_prediction(self, predicted_categories, extracted_skills):
        """
        Refine ML model prediction using your skill-based rules

        Args:
            predicted_categories: List of predictions from ML model
            extracted_skills: List of skills from resume

        Returns:
            Refined list of category predictions
        """
        # Get rule-based category from your data
        rule_based_category, rule_confidence = self.get_category_from_skills(extracted_skills)

        if not rule_based_category:
            return predicted_categories

        # Check if rule-based category is in top predictions
        predicted_names = [p['category'] for p in predicted_categories]

        if rule_based_category not in predicted_names:
            # Add rule-based category if it has good confidence
            if rule_confidence > 0.3:
                predicted_categories.insert(0, {
                    'category': rule_based_category,
                    'confidence': rule_confidence,
                    'confidence_percentage': f'{rule_confidence * 100:.1f}%',
                    'source': 'skill-matching'
                })
        else:
            # Boost confidence of matching category
            for pred in predicted_categories:
                if pred['category'] == rule_based_category:
                    # Combine ML confidence with rule-based confidence
                    combined_confidence = (pred['confidence'] + rule_confidence) / 2
                    pred['confidence'] = min(combined_confidence * 1.2, 0.99)
                    pred['confidence_percentage'] = f"{pred['confidence'] * 100:.1f}%"
                    pred['source'] = 'ml+skill-matching'

        # Re-sort by confidence
        predicted_categories.sort(key=lambda x: x['confidence'], reverse=True)

        return predicted_categories[:3]  # Keep top 3


# Helper function to use with existing code
def refine_category_prediction(predicted_categories, extracted_skills, category_skills_map):
    """
    Helper function for backward compatibility

    Args:
        predicted_categories: List of ML predictions
        extracted_skills: List of extracted skills
        category_skills_map: Your category->skills mapping dict

    Returns:
        Refined predictions
    """
    mapper = CategoryMapper(category_skills_map)
    return mapper.refine_category_prediction(predicted_categories, extracted_skills)