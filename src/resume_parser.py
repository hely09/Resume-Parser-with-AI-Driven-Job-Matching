"""
Resume Parser - Extracts information from PDF resumes
"""
import re
import spacy
from spacy.matcher import Matcher
from pdfminer.high_level import extract_text

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        return extract_text(pdf_path)
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def extract_contact_number(text):
    """Extract contact number using regex"""
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    return match.group() if match else None


def extract_email(text):
    """Extract email using regex"""
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    return match.group() if match else None


def extract_name(resume_text):
    """Extract name using spaCy"""
    matcher = Matcher(nlp.vocab)
    patterns = [
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}],
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}],
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}]
    ]
    for pattern in patterns:
        matcher.add('NAME', patterns=[pattern])

    doc = nlp(resume_text)
    matches = matcher(doc)

    for match_id, start, end in matches:
        span = doc[start:end]
        if span.start_char < 2000:
            return span.text
    return None


def extract_education(text):
    """Extract education information"""
    education = []
    pattern = r"(?i)(?:Bsc|\bB\.\w+|\bM\.\w+|\bPh\.D\.?\w*|\bBachelor(?:'s)?(?:\s+of)?|\bMaster(?:'s)?(?:\s+of)?|\bPh\.?D\.?)\s(?:\w+\s){0,5}?\w+"
    matches = re.findall(pattern, text)
    for match in matches:
        education.append(match.strip())
    return education


def _is_valid_skill(skill_text: str) -> bool:
    """
    Validate if extracted text is actually a skill
    """
    skill_lower = skill_text.lower().strip()

    # Too short or too long
    if len(skill_text) < 2 or len(skill_text) > 30:
        return False

    # Contains numbers (likely dates or addresses)
    if any(char.isdigit() for char in skill_text):
        return False

    # Common non-skill words and phrases
    exclude_phrases = {
        # Locations
        'ahmedabad', 'gujarat', 'india', 'mumbai', 'delhi', 'bangalore',
        # Time/Date
        'present', 'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        # Job titles/roles (too generic)
        'intern', 'interns', 'internship', 'developer', 'engineer', 'analyst',
        'scientist', 'manager', 'specialist', 'consultant', 'architect',
        # Actions/Verbs
        'work', 'working', 'employed', 'focusing', 'aligning', 'construct',
        'present', 'effectively', 'rigorous', 'using', 'with', 'for',
        'processing', 'testing', 'validation', 'development', 'forecasting',
        # Generic terms
        'insights', 'objectives', 'individuals', 'services', 'business',
        'accurate', 'backend', 'frameworks', 'the', 'and', 'to', 'on',
        'pvt', 'ltd', 'company', 'technologies', 'experience',
        # Company names patterns
        'infolabz', 'technishal', 'pvt ltd', 'services',
        # Project descriptions
        'weather app', 'app using', 'projects', 'project',
        'data engineering', 'model development', 'data scientist',
        'data structures'
    }

    # Check if it matches excluded phrases
    if skill_lower in exclude_phrases:
        return False

    # Check if it contains excluded words
    words = skill_lower.split()
    if any(word in exclude_phrases for word in words):
        return False

    # Must contain at least one letter
    if not any(c.isalpha() for c in skill_text):
        return False

    # Check if it's a sentence (contains too many words)
    if len(words) > 4:
        return False

    # Reject if it contains common stop words
    stop_words = {'the', 'and', 'for', 'with', 'using', 'to', 'on', 'in', 'at'}
    if any(word in stop_words for word in words):
        return False

    return True


def _clean_skill_token(tok: str) -> str:
    """Clean individual skill tokens"""
    tok = tok.strip()
    tok = re.sub(r"^[\-\u2022\•\*\)\(\[\]\d\.]+\s*", "", tok)
    tok = re.sub(r"[\(\)\[\]\.]+$", "", tok)
    tok = tok.strip(" .,-;:/|")
    return tok


def extract_skills(text: str, debug=False) -> list:
    """
    Extract skills from resume text - IMPROVED VERSION
    Only extracts from Skills section, not from entire resume
    """
    if debug:
        print("\n" + "="*70)
        print("DEBUG: Searching for skills sections")
        print("="*70)

    lines = text.split('\n')
    skills_content = []
    in_skills_section = False

    # Skills section headers - MORE FLEXIBLE MATCHING
    skill_headers = [
        'technical skills', 'core skills', 'key skills', 'skills',
        'technical competencies', 'technologies', 'programming languages',
        'tech stack', 'technical stack', 'it skills', 'professional skills',
        'skill', 'competencies', 'technology'
    ]

    # Section headers that indicate end of skills - MUST BE CLEAR HEADERS
    other_headers = [
        'experience', 'work experience', 'professional experience',
        'education', 'projects', 'project', 'certifications', 'summary',
        'objective', 'internship', 'employment', 'achievements',
        'work history', 'employment history', 'career history',
        'academic', 'qualification', 'training', 'courses'
    ]

    for i, line in enumerate(lines):
        line_clean = line.strip().lower()

        # Skip empty lines
        if not line_clean:
            continue

        # Check if this line is a skills header (exact match or starts with)
        is_skill_header = False
        for header in skill_headers:
            # Match if line starts with header, or is exactly the header
            if line_clean == header or line_clean.startswith(header + ':') or line_clean.startswith(header + ' '):
                is_skill_header = True
                break

        # Check if this is another section header (must be at start of line and clear)
        is_other_header = False
        if in_skills_section:  # Only check for end headers if we're in skills section
            for header in other_headers:
                # Must be a clear section header - at start and not just a word in content
                if (line_clean == header or
                    line_clean.startswith(header + ':') or
                    (line_clean.startswith(header + ' ') and len(line_clean.split()) <= 3)):
                    # Additional check: line should not be too long (likely not a header if > 50 chars)
                    if len(line_clean) <= 50:
                        is_other_header = True
                        break

        if is_skill_header:
            in_skills_section = True
            if debug:
                print(f"\n✓ SKILLS SECTION FOUND at line {i}: '{line.strip()}'")

            # Extract skills from same line after the header
            for header in skill_headers:
                if line_clean.startswith(header):
                    # Get remaining text after header
                    remaining = line[len(header):].strip()
                    remaining = re.sub(r'^[::\-\•]+\s*', '', remaining)
                    if remaining and len(remaining) > 2:  # Make sure it's not just punctuation
                        skills_content.append(remaining)
            continue

        if is_other_header:
            if debug:
                print(f"\n✗ SKILLS SECTION ENDED at line {i}: '{line.strip()}'")
            in_skills_section = False
            break  # Stop looking for skills after section ends

        if in_skills_section:
            # Add content from skills section
            skills_content.append(line.strip())

    if not skills_content:
        if debug:
            print("\n❌ No skills section found!")
        return []

    if debug:
        print(f"\n📝 Skills section content ({len(skills_content)} lines):")
        for idx, content in enumerate(skills_content[:5]):  # Show first 5 lines
            print(f"  {idx+1}. {content[:80]}")

    # Parse skills from content
    all_skills = []

    for content in skills_content:
        # Skip if line is too short
        if len(content.strip()) < 2:
            continue

        # Split by common delimiters
        parts = []
        if ',' in content:
            parts = content.split(',')
        elif '|' in content:
            parts = content.split('|')
        elif ';' in content:
            parts = content.split(';')
        elif '•' in content or '●' in content or '○' in content:
            parts = re.split(r'[•●○]', content)
        elif '\t' in content:  # Tab-separated
            parts = content.split('\t')
        else:
            # Try to split by multiple spaces (2 or more)
            parts = re.split(r'\s{2,}', content)

            # If no multi-space splits, treat whole line as one skill
            if len(parts) == 1:
                parts = [content]

        for part in parts:
            cleaned = _clean_skill_token(part)
            if cleaned and len(cleaned) > 1:
                all_skills.append(cleaned)

    # Filter and validate skills
    final_skills = []
    seen = set()

    for skill in all_skills:
        skill_normalized = skill.lower().strip()

        # Skip if already seen
        if skill_normalized in seen:
            continue

        # Validate skill
        if _is_valid_skill(skill):
            seen.add(skill_normalized)
            final_skills.append(skill.title())

    if debug:
        print(f"\n✓ Extracted {len(final_skills)} valid skills")
        print(f"Skills: {final_skills[:20]}")  # Show first 20
        if len(final_skills) > 20:
            print(f"... and {len(final_skills) - 20} more")

    return final_skills


def parse_resume(pdf_path, debug=False):
    """
    Complete resume parsing
    Returns dict with all extracted information
    """
    text = extract_text_from_pdf(pdf_path)

    return {
        'name': extract_name(text),
        'email': extract_email(text),
        'contact': extract_contact_number(text),
        'education': extract_education(text),
        'skills': extract_skills(text, debug=debug),
        'raw_text': text
    }