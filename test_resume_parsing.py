"""
Test resume parsing to verify skills extraction
Dynamically loads expected skills from your trained model
"""
from src.resume_parser import parse_resume
from src.model_trainer import load_trained_models
from pathlib import Path


def test_resume_parsing(resume_path, expected_category=None):
    """
    Test resume parsing and compare with your dataset
    """

    print("=" * 70)
    print("TESTING RESUME PARSING")
    print("=" * 70)

    # Load your trained model data
    print("\n→ Loading your category-skills data...")
    try:
        model_data = load_trained_models()
        category_skills_map = model_data['category_skills_map']
        print(f"✓ Loaded {len(category_skills_map)} categories from your data")
        print(f"  Categories: {list(category_skills_map.keys())}")
    except Exception as e:
        print(f"❌ Error loading model data: {e}")
        print("⚠ Make sure you've run: python train_model.py")
        return

    # Parse resume with debug mode
    print(f"\n→ Parsing resume: {resume_path}")
    result = parse_resume(resume_path, debug=True)

    print("\n" + "=" * 70)
    print("PARSING RESULTS")
    print("=" * 70)

    print(f"\nName: {result['name']}")
    print(f"Email: {result['email']}")
    print(f"Contact: {result['contact']}")

    if result['education']:
        print(f"Education:")
        for edu in result['education']:
            print(f"  • {edu}")

    print(f"\n📊 Extracted Skills: {len(result['skills'])} total")

    if not result['skills']:
        print("\n❌ NO SKILLS EXTRACTED!")
        print("\nPossible reasons:")
        print("  1. No 'Skills' section found in resume")
        print("  2. Skills section has unusual format")
        print("  3. Skills are embedded in experience/projects")
        return result

    # Analyze which categories match
    print("\n" + "=" * 70)
    print("SKILL DISTRIBUTION ACROSS YOUR CATEGORIES")
    print("=" * 70)

    category_matches = {}
    skill_category_map = {}

    for skill in result['skills']:
        skill_lower = skill.lower()
        matched_categories = []

        for category, skills_list in category_skills_map.items():
            if skill_lower in [s.lower() for s in skills_list]:
                matched_categories.append(category)
                category_matches[category] = category_matches.get(category, 0) + 1

        skill_category_map[skill] = matched_categories

    if category_matches:
        sorted_matches = sorted(category_matches.items(), key=lambda x: x[1], reverse=True)

        print("\nYour skills match these categories:")
        for category, count in sorted_matches:
            percentage = count / len(result['skills']) * 100
            bar = "█" * int(percentage / 5)
            print(f"  {category:25s}: {count:2d} skills ({percentage:5.1f}%) {bar}")

        print(f"\n💡 Predicted Category: {sorted_matches[0][0]}")
        print(f"   Confidence: {sorted_matches[0][1]}/{len(result['skills'])} skills matched")
    else:
        print("\n⚠ No skills matched any category in your dataset!")
        print("\nExtracted skills:")
        for skill in result['skills']:
            print(f"  • {skill}")
        print("\n💡 These skills might not be in your training dataset.")

    # Show detailed skill matching
    if expected_category and expected_category in category_skills_map:
        print("\n" + "=" * 70)
        print(f"DETAILED COMPARISON WITH '{expected_category}'")
        print("=" * 70)

        expected_skills = set(s.lower() for s in category_skills_map[expected_category])
        extracted_skills = set(s.lower() for s in result['skills'])

        matched = extracted_skills.intersection(expected_skills)
        missed = expected_skills - extracted_skills

        if matched:
            print(f"\n✓ Matched Skills ({len(matched)}):")
            for skill in sorted(matched)[:20]:
                print(f"  ✓ {skill}")
            if len(matched) > 20:
                print(f"  ... and {len(matched) - 20} more")

        if missed:
            print(f"\n❌ Skills in {expected_category} Dataset But Not in Resume ({len(missed)}):")
            for skill in sorted(missed)[:10]:
                print(f"  ❌ {skill}")
            if len(missed) > 10:
                print(f"  ... and {len(missed) - 10} more")

        coverage = len(matched) / len(expected_skills) * 100 if expected_skills else 0
        print(f"\n📈 Coverage: {coverage:.1f}% of '{expected_category}' skills")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        resume_path = sys.argv[1]
    else:
        resume_path = "uploads/Krisha Doshi Resume.pdf"  # Default

    expected_category = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(resume_path).exists():
        print(f"❌ Error: Resume not found: {resume_path}")
        print("\nUsage:")
        print(f"  python test_resume_parsing.py <resume_path> [category]")
        print(f"\nExamples:")
        print(f"  python test_resume_parsing.py uploads/myresume.pdf")
        print(f"  python test_resume_parsing.py uploads/myresume.pdf 'Software Engineer'")
        sys.exit(1)

    test_resume_parsing(resume_path, expected_category)