"""
Skill Matcher - Semantic and hybrid skill matching
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import SEMANTIC_MODEL_NAME, SEMANTIC_THRESHOLD


class SemanticSkillMatcher:
    """Uses sentence transformers for semantic skill matching"""

    def __init__(self, model_name=SEMANTIC_MODEL_NAME):
        print(f"→ Loading semantic model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.category_embeddings = {}
        self.category_skills_map = {}
        print("✓ Semantic model loaded")

    def set_category_skills(self, category_skills_map):
        """Set category-skills mapping"""
        self.category_skills_map = category_skills_map
        print(f"✓ Loaded {len(self.category_skills_map)} categories")

    def precompute_embeddings(self):
        """Precompute embeddings for all skills"""
        print("\n→ Precomputing skill embeddings...")

        for category, skills in self.category_skills_map.items():
            embeddings = self.model.encode(skills, show_progress_bar=False)
            self.category_embeddings[category] = {
                'skills': skills,
                'embeddings': embeddings
            }

        print(f"✓ Precomputed embeddings for {len(self.category_embeddings)} categories")

    def find_semantic_matches(self, resume_skill, category, threshold=SEMANTIC_THRESHOLD):
        """Find semantically similar skills in a category"""
        if category not in self.category_embeddings:
            return []

        resume_embedding = self.model.encode([resume_skill], show_progress_bar=False)
        category_data = self.category_embeddings[category]
        category_skills = category_data['skills']
        category_embeds = category_data['embeddings']

        similarities = cosine_similarity(resume_embedding, category_embeds)[0]

        matches = []
        for skill, similarity in zip(category_skills, similarities):
            if similarity >= threshold:
                matches.append((skill, float(similarity)))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def match_resume_skills_to_category(self, resume_skills, category, threshold=SEMANTIC_THRESHOLD):
        """Match all resume skills to a category"""
        matched_skills = {}
        covered_category_skills = set()

        for resume_skill in resume_skills:
            matches = self.find_semantic_matches(resume_skill, category, threshold)
            if matches:
                matched_skills[resume_skill] = matches
                for matched_skill, _ in matches:
                    covered_category_skills.add(matched_skill.lower())

        return matched_skills, covered_category_skills


class HybridSkillMatcher:
    """Combines exact matching and semantic matching"""

    def __init__(self, semantic_matcher, category_skills_map):
        self.semantic_matcher = semantic_matcher
        self.category_skills_map = category_skills_map

    def get_matched_skills(self, resume_skills, category, semantic_threshold=SEMANTIC_THRESHOLD):
        """Get all matched skills using both exact and semantic matching"""
        category_skills_lower = set([s.lower().strip() for s in self.category_skills_map[category]])
        resume_skills_lower = [s.lower().strip() for s in resume_skills]

        # Exact matches
        exact_matches = list(category_skills_lower.intersection(resume_skills_lower))

        # Semantic matches
        semantic_matches, covered_semantic = self.semantic_matcher.match_resume_skills_to_category(
            resume_skills, category, semantic_threshold
        )

        # Combine
        all_covered = set(exact_matches).union(covered_semantic)

        return {
            'exact_matches': exact_matches,
            'semantic_matches': semantic_matches,
            'all_covered_skills': all_covered,
            'total_matched': len(all_covered)
        }

    def get_missing_skills(self, resume_skills, category, semantic_threshold=SEMANTIC_THRESHOLD):
        """Compute missing skills"""
        category_skills_lower = set([s.lower().strip() for s in self.category_skills_map[category]])
        matched = self.get_matched_skills(resume_skills, category, semantic_threshold)
        all_covered = matched['all_covered_skills']
        missing = sorted(list(category_skills_lower - all_covered))
        return missing

    def get_skill_coverage_report(self, resume_skills, category, semantic_threshold=SEMANTIC_THRESHOLD):
        """Generate detailed skill coverage report"""
        matched = self.get_matched_skills(resume_skills, category, semantic_threshold)
        missing = self.get_missing_skills(resume_skills, category, semantic_threshold)

        total_required = len(self.category_skills_map[category])
        total_matched = matched['total_matched']
        coverage_percentage = (total_matched / total_required * 100) if total_required > 0 else 0

        return {
            'category': category,
            'total_required_skills': total_required,
            'total_matched_skills': total_matched,
            'coverage_percentage': coverage_percentage,
            'exact_matches': matched['exact_matches'],
            'semantic_matches': matched['semantic_matches'],
            'missing_skills': missing
        }