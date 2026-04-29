"""
Streamlit App - AI Resume Analyzer with Job Description Matching
"""
import streamlit as st
from pathlib import Path
import plotly.graph_objects as go
import json
import re

from src.config import *
from src.analyzer import ResumeAnalyzer

# Page config
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .skill-badge {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        margin: 0.25rem;
        font-size: 0.9rem;
    }
    .missing-skill-badge {
        display: inline-block;
        background-color: #ffebee;
        color: #c62828;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        margin: 0.25rem;
        font-size: 0.9rem;
    }
    .matched-skill-badge {
        display: inline-block;
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        margin: 0.25rem;
        font-size: 0.9rem;
    }
    .jd-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_analyzer():
    """Load analyzer (cached)"""
    return ResumeAnalyzer(courses_csv_path=str(COURSES_CSV))


def extract_jd_skills(jd_text, analyzer):
    """
    Extract ONLY technical skills from job description
    Uses strict filtering to avoid generic words
    """
    import spacy

    # Load spaCy model if not already loaded
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    jd_text_lower = jd_text.lower()

    # Collect all known technical skills from all categories
    all_known_skills = set()
    for category, skills in analyzer.category_skills_map.items():
        all_known_skills.update([s.lower().strip() for s in skills])

    # Words to EXCLUDE (common non-technical words)
    exclude_words = {
        # Generic action words
        'abilities', 'ability', 'experience', 'familiarity', 'knowledge',
        'understanding', 'skills', 'skill', 'working', 'development', 'developer',
        'engineering', 'engineer', 'architecture', 'architect', 'design', 'designer',

        # Soft skills / attributes
        'communication', 'excellent', 'strong', 'good', 'solid', 'proven',
        'attention', 'detail', 'team', 'collaboration', 'problem', 'solving',
        'analytical', 'critical', 'thinking', 'leadership', 'management',

        # Methodologies (too generic)
        'agile', 'scrum', 'waterfall', 'kanban', 'devops', 'ci/cd', 'cicd',

        # Generic tech terms
        'cloud', 'databases', 'database', 'back', 'front', 'full', 'stack',
        'web', 'mobile', 'application', 'applications', 'system', 'systems',
        'software', 'hardware', 'network', 'networking', 'security',
        'testing', 'deployment', 'integration', 'implementation',

        # Job requirements language
        'required', 'preferred', 'must', 'should', 'need', 'needs',
        'looking', 'seeking', 'candidate', 'candidates', 'position',
        'role', 'responsibilities', 'requirements', 'qualifications',

        # Time/Experience
        'years', 'year', 'experience', 'senior', 'junior', 'mid', 'level',

        # Common verbs
        'using', 'use', 'work', 'develop', 'build', 'create', 'implement',
        'maintain', 'support', 'manage', 'lead', 'collaborate', 'design',

        # Education
        'degree', 'bachelor', 'master', 'phd', 'certification', 'certified',

        # Location/Company
        'remote', 'office', 'location', 'based', 'company', 'team', 'department',

        # Generic IT terms (too broad)
        'api', 'apis', 'rest', 'restful', 'microservices', 'service', 'services',
        'framework', 'frameworks', 'library', 'libraries', 'tool', 'tools',
        'platform', 'platforms', 'technology', 'technologies', 'solution', 'solutions',

        # Programming concepts (too generic)
        'oop', 'object', 'oriented', 'programming', 'paradigm', 'patterns',
        'algorithms', 'structures', 'core', 'fundamentals', 'basics',

        # Version control (too generic)
        'version', 'control', 'source', 'code', 'repository', 'repositories',

        # Containers (keep specific tools, remove generic)
        'containerization', 'containers', 'container', 'orchestration','eg','e.g.',

        # Monitoring/Logging (too generic)
        'monitoring', 'logging', 'debugging', 'troubleshooting',

        # Documentation
        'documentation', 'documents', 'writing', 'technical'
    }

    # Specific TECHNICAL SKILLS to look for (curated list)
    priority_tech_skills = {
        # Programming Languages
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby',
        'php', 'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab',

        # Web Frameworks
        'react', 'reactjs', 'react.js', 'angular', 'vue', 'vuejs', 'vue.js',
        'nodejs', 'node.js', 'django', 'flask', 'fastapi', 'spring', 'springboot',
        'express', 'expressjs', 'asp.net', 'laravel', 'rails', 'nextjs', 'next.js',

        # Databases
        'sql', 'mysql', 'postgresql', 'postgres', 'mongodb', 'redis',
        'cassandra', 'elasticsearch', 'dynamodb', 'oracle', 'mariadb',
        'sqlite', 'neo4j', 'couchdb', 'influxdb',

        # Cloud Platforms (specific services)
        'aws', 'azure', 'gcp', 'ec2', 's3', 'lambda', 'cloudformation',
        'azure devops', 'google cloud', 'heroku', 'digitalocean',

        # DevOps Tools (specific)
        'docker', 'kubernetes', 'k8s', 'jenkins', 'terraform', 'ansible',
        'gitlab', 'github actions', 'circleci', 'travis ci',

        # Version Control (specific tools)
        'git', 'github', 'gitlab', 'bitbucket', 'svn',

        # Data Science / ML
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn',
        'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly',
        'jupyter', 'spark', 'pyspark', 'hadoop', 'kafka', 'airflow',

        # Frontend
        'html', 'html5', 'css', 'css3', 'sass', 'scss', 'less',
        'bootstrap', 'tailwind', 'material-ui', 'mui', 'webpack', 'vite',

        # Mobile
        'react native', 'flutter', 'android', 'ios', 'xamarin',

        # Testing
        'jest', 'mocha', 'pytest', 'junit', 'selenium', 'cypress',

        # Data Visualization
        'tableau', 'power bi', 'powerbi', 'looker', 'qlik', 'd3.js', 'd3',

        # Message Queues
        'rabbitmq', 'kafka', 'redis', 'celery',

        # Web Servers
        'nginx', 'apache', 'tomcat', 'iis',

        # Operating Systems
        'linux', 'unix', 'ubuntu', 'centos', 'windows', 'macos',

        # Shell/Scripting
        'bash', 'shell', 'powershell', 'cmd',

        # GraphQL
        'graphql', 'apollo',

        # IDEs (specific)
        'vscode', 'pycharm', 'intellij', 'eclipse', 'visual studio'
    }

    found_skills = set()

    # Method 1: Check against known skills from category map (STRICT)
    for skill in all_known_skills:
        # Skip if it's in exclude list
        if skill in exclude_words:
            continue

        # Must be at least 2 characters
        if len(skill) < 2:
            continue

        # Create pattern with word boundaries
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, jd_text_lower):
            found_skills.add(skill)

    # Method 2: Check priority technical skills
    for skill in priority_tech_skills:
        if skill in exclude_words:
            continue

        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, jd_text_lower):
            found_skills.add(skill)

    # Method 3: Use spaCy to extract potential tech terms (VERY STRICT)
    doc = nlp(jd_text)
    for token in doc:
        word = token.text.lower().strip()

        # Skip if too short or in exclude list
        if len(word) < 2 or word in exclude_words:
            continue

        # Only consider if it's:
        # 1. A proper noun (likely a technology name) OR
        # 2. Contains special chars like . # + (react.js, c#, c++)
        if (token.pos_ == 'PROPN' or '.' in word or '#' in word or '+' in word):
            # Additional validation
            if not word.isalpha() or word.isupper() or '.' in word or '#' in word or '+' in word:
                # Check it's not a common word
                if word not in exclude_words:
                    found_skills.add(word)

    # Final filtering: Remove any skills that are too generic or single letters
    final_skills = []
    for skill in found_skills:
        # Remove single characters (except valid ones like 'r', 'c')
        if len(skill) == 1 and skill not in ['r', 'c']:
            continue

        # Remove if it's a common English word (unless it's a known tech term)
        if skill in exclude_words:
            continue

        # Keep it
        final_skills.append(skill)

    # Remove duplicates and sort
    final_skills = sorted(list(set(final_skills)))

    return final_skills


def display_basic_info(analysis):
    """Display basic information section"""
    st.markdown('<div class="sub-header">📋 Basic Information</div>', unsafe_allow_html=True)

    basic_info = analysis['basic_info']

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Name:**")
        st.write(basic_info['name'] or "Not found")

    with col2:
        st.markdown("**Email:**")
        st.write(basic_info['email'] or "Not found")

    with col3:
        st.markdown("**Contact:**")
        st.write(basic_info['contact'] or "Not found")

    if basic_info['education']:
        st.markdown("**Education:**")
        for edu in basic_info['education']:
            st.write(f"• {edu}")


def display_extracted_skills(analysis):
    """Display extracted skills"""
    st.markdown('<div class="sub-header">🛠️ Extracted Skills</div>', unsafe_allow_html=True)

    skills = sorted(analysis['skills']['extracted_skills'])
    st.info(f"**Total Skills Found:** {len(skills)}")

    skills_html = " ".join([f'<span class="skill-badge">{skill}</span>' for skill in skills])
    st.markdown(skills_html, unsafe_allow_html=True)


def display_category_prediction(analysis):
    """Display category predictions"""
    st.markdown('<div class="sub-header">🎯 Predicted Job Categories</div>', unsafe_allow_html=True)

    predictions = analysis['category_prediction']['top_predictions']

    categories = [p['category'] for p in predictions]
    confidences = [p['confidence'] * 100 for p in predictions]

    fig = go.Figure(data=[
        go.Bar(
            x=confidences,
            y=categories,
            orientation='h',
            marker=dict(color=confidences, colorscale='Blues'),
            text=[f"{c:.1f}%" for c in confidences],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Category Prediction Confidence",
        xaxis_title="Confidence (%)",
        yaxis_title="Category",
        height=300,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def display_skill_analysis(analysis):
    """Display skill match analysis"""
    st.markdown('<div class="sub-header">📊 Skill Match Analysis</div>', unsafe_allow_html=True)

    skill_analysis = analysis['skill_analysis']

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Category", skill_analysis['category'])

    with col2:
        st.metric("Required Skills", skill_analysis['total_required_skills'])

    with col3:
        st.metric("Matched Skills", skill_analysis['total_matched_skills'])

    with col4:
        coverage = skill_analysis['coverage_percentage']
        st.metric("Coverage", f"{coverage:.1f}%")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=coverage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Skill Coverage"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': SUCCESS_COLOR if coverage >= 70 else WARNING_COLOR},
            'steps': [
                {'range': [0, 50], 'color': "#ffebee"},
                {'range': [50, 70], 'color': "#fff3e0"},
                {'range': [70, 100], 'color': "#e8f5e9"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))

    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**✅ Exact Matches:**")
    exact_matches = sorted(skill_analysis['exact_matches'][:15])
    if exact_matches:
        matches_html = " ".join([f'<span class="skill-badge">{skill}</span>' for skill in exact_matches])
        st.markdown(matches_html, unsafe_allow_html=True)
        if len(skill_analysis['exact_matches']) > 15:
            st.write(f"... and {len(skill_analysis['exact_matches']) - 15} more")
    else:
        st.write("None")

    if skill_analysis['semantic_matches']:
        st.markdown("**🔗 Semantic Matches (Similar Skills):**")
        for resume_skill, matches in list(skill_analysis['semantic_matches'].items())[:5]:
            with st.expander(f"'{resume_skill}' matched to:"):
                for matched_skill, score in matches[:3]:
                    st.write(f"• {matched_skill} (similarity: {score:.2f})")


def display_missing_skills(analysis):
    """Display missing skills"""
    st.markdown('<div class="sub-header">❌ Missing Skills</div>', unsafe_allow_html=True)

    missing_skills = analysis['skill_analysis']['missing_skills']

    st.warning(f"**Total Missing Skills:** {len(missing_skills)}")

    display_skills = missing_skills[:30]
    skills_html = " ".join([f'<span class="missing-skill-badge">{skill}</span>' for skill in display_skills])
    st.markdown(skills_html, unsafe_allow_html=True)

    if len(missing_skills) > 30:
        st.write(f"... and {len(missing_skills) - 30} more skills")


def display_course_recommendations(analysis):
    """Display course recommendations"""
    st.markdown('<div class="sub-header">📚 Course Recommendations</div>', unsafe_allow_html=True)

    recommendations = analysis['course_recommendations']

    st.info(f"Showing recommendations for top {len(recommendations)} priority skills")

    for i, rec in enumerate(recommendations[:10], 1):
        with st.expander(f"🎯 {i}. {rec['skill'].upper()}", expanded=(i <= 1)):
            for course in rec['courses']:
                st.markdown(f"**{course['course_name']}**")
                st.write(f"Platform: {course['platform']}")
                if 'url' in course:
                    st.markdown(f"[View Course]({course['url']})")
                st.markdown("---")


def display_jd_comparison(resume_skills, jd_skills, jd_text, analyzer):
    """Display JD comparison analysis with semantic matching"""
    st.markdown('<div class="sub-header">🎯 Job Description Match Analysis</div>', unsafe_allow_html=True)

    # Convert to sets for comparison
    resume_skills_set = set([s.lower().strip() for s in resume_skills])
    jd_skills_set = set([s.lower().strip() for s in jd_skills])

    # Exact matches
    exact_matched_skills = resume_skills_set.intersection(jd_skills_set)

    # Semantic matching for remaining JD skills
    semantic_matches = {}
    unmatched_jd_skills = jd_skills_set - exact_matched_skills

    for jd_skill in unmatched_jd_skills:
        # Find semantically similar resume skills
        matches = analyzer.semantic_matcher.model.encode([jd_skill])
        resume_embeddings = analyzer.semantic_matcher.model.encode(list(resume_skills))

        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(matches, resume_embeddings)[0]

        # Get top matches above threshold
        top_matches = []
        for idx, sim in enumerate(similarities):
            if sim >= 0.65:  # Threshold for semantic similarity
                top_matches.append((resume_skills[idx], float(sim)))

        if top_matches:
            top_matches.sort(key=lambda x: x[1], reverse=True)
            semantic_matches[jd_skill] = top_matches[:3]

    # Calculate coverage
    semantically_matched = len(semantic_matches)
    total_matched = len(exact_matched_skills) + semantically_matched
    missing_from_jd = len(jd_skills_set) - total_matched

    if len(jd_skills_set) > 0:
        match_percentage = (total_matched / len(jd_skills_set)) * 100
    else:
        match_percentage = 0

    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("JD Skills", len(jd_skills_set))

    with col2:
        st.metric("Exact Matches", len(exact_matched_skills))

    with col3:
        st.metric("Semantic Matches", semantically_matched)

    with col4:
        st.metric("Missing", missing_from_jd)

    with col5:
        st.metric("Match Score", f"{match_percentage:.1f}%")

    # Match score gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=match_percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "JD Match Score"},
        delta={'reference': 70},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "#ffcdd2"},
                {'range': [40, 70], 'color': "#fff9c4"},
                {'range': [70, 100], 'color': "#c8e6c9"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))

    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**✅ Exact Matches:**")
        if exact_matched_skills:
            matched_html = " ".join([f'<span class="matched-skill-badge">{skill}</span>'
                                    for skill in sorted(exact_matched_skills)[:20]])
            st.markdown(matched_html, unsafe_allow_html=True)
            if len(exact_matched_skills) > 20:
                st.write(f"... and {len(exact_matched_skills) - 20} more")
        else:
            st.write("No exact matches found")

        # Semantic matches
        if semantic_matches:
            st.markdown("**🔗 Semantic Matches:**")
            for jd_skill, matches in list(semantic_matches.items())[:5]:
                with st.expander(f"JD: '{jd_skill}' → Resume skills:"):
                    for resume_skill, score in matches:
                        st.write(f"• {resume_skill} (similarity: {score:.2f})")

    with col2:
        remaining_missing = jd_skills_set - exact_matched_skills - set(semantic_matches.keys())
        st.markdown("**❌ Missing from Resume:**")
        if remaining_missing:
            missing_html = " ".join([f'<span class="missing-skill-badge">{skill}</span>'
                                    for skill in sorted(remaining_missing)[:20]])
            st.markdown(missing_html, unsafe_allow_html=True)
            if len(remaining_missing) > 20:
                st.write(f"... and {len(remaining_missing) - 20} more")
        else:
            st.success("All JD skills covered!")

    # Insights
    st.markdown("---")
    if match_percentage >= 70:
        st.success("🎉 **Excellent Match!** Your resume aligns well with the job requirements.")
    elif match_percentage >= 50:
        st.warning("⚠️ **Moderate Match.** Consider highlighting or adding missing skills if you have them.")
    else:
        st.error("❗ **Low Match.** Significant skill gap detected. Consider upskilling or tailoring your resume.")


def main():
    # Header
    st.markdown('<div class="main-header">📄 AI Resume Analyzer</div>', unsafe_allow_html=True)
    st.markdown("Upload your resume to get detailed analysis, skill matching, and course recommendations!")

    # Sidebar
    with st.sidebar:
        st.image("logo2.png",
                width=100)
        # st.markdown("### About")
        st.write("🚀 **Transform your job search with AI!**")
        # st.write("")
        # st.write("**What you'll get:**")
        st.write("• Smart skill extraction from your resume")
        st.write("• AI-powered job category predictions")
        st.write("• Personalized course recommendations")
        st.write("• Job description match scoring")
        st.write("• Gap analysis with actionable insights")

        st.markdown("---")
        st.markdown("### Settings")

        # Fixed threshold - no slider
        semantic_threshold = 0.65  # Fixed value from config

        debug_mode = st.checkbox("Debug Mode", help="Show detailed skill extraction info")

    # Check if models are trained
    if not TRAINED_MODEL_PATH.exists():
        st.error("⚠️ Models not found! Please run `python train_model.py` first.")
        st.stop()

    # Load analyzer
    with st.spinner("Loading AI models..."):
        analyzer = load_analyzer()

    st.success("✅ System ready!")

    # File upload section
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "📄 Upload Resume (PDF) *Required",
            type=['pdf'],
            help="Upload your resume in PDF format"
        )

    with col2:
        st.markdown("### 📋 Job Description (Optional)")
        use_jd = st.checkbox("Compare with Job Description", value=False)

    # JD text area
    jd_text = None
    if use_jd:
        st.markdown('<div class="jd-section">', unsafe_allow_html=True)
        st.markdown("**Paste the Job Description below:**")
        jd_text = st.text_area(
            "Job Description",
            height=200,
            placeholder="Paste the complete job description here...\n\nWe are looking for a Senior Python Developer with experience in Django, AWS, Docker...",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        # Save uploaded file
        save_path = UPLOADS_DIR / uploaded_file.name
        with open(save_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        st.info(f"📄 File loaded: {uploaded_file.name}")

        # Analyze button
        if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
            with st.spinner("Analyzing resume... This may take a few moments."):
                try:
                    analysis = analyzer.analyze_resume(
                        str(save_path),
                        semantic_threshold=semantic_threshold,
                        debug=debug_mode
                    )

                    if analysis:
                        st.success("✅ Analysis complete!")

                        # Extract JD skills if provided
                        jd_skills = []
                        if use_jd and jd_text:
                            jd_skills = extract_jd_skills(jd_text, analyzer)

                        # Store in session state
                        st.session_state['analysis'] = analysis
                        st.session_state['jd_text'] = jd_text
                        st.session_state['jd_skills'] = jd_skills
                        st.session_state['use_jd'] = use_jd

                    else:
                        st.error("❌ Analysis failed. Please check your resume format.")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

        # Display results if available
        if 'analysis' in st.session_state:
            analysis = st.session_state['analysis']
            stored_jd = st.session_state.get('jd_text')
            stored_jd_skills = st.session_state.get('jd_skills', [])
            stored_use_jd = st.session_state.get('use_jd', False)

            st.markdown("---")

            # Create tabs based on whether JD is used
            if stored_use_jd and stored_jd and stored_jd_skills:
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📋 Resume Overview",
                    "🎯 JD Analysis",
                    "📊 Skill Analysis",
                    "❌ Missing Skills",
                    "📚 Courses"
                ])

                with tab1:
                    display_basic_info(analysis)
                    st.markdown("---")
                    display_extracted_skills(analysis)
                    st.markdown("---")
                    display_category_prediction(analysis)

                with tab2:
                    resume_skills = analysis['skills']['extracted_skills']
                    display_jd_comparison(resume_skills, stored_jd_skills, stored_jd, analyzer)

                with tab3:
                    display_skill_analysis(analysis)

                with tab4:
                    display_missing_skills(analysis)

                with tab5:
                    display_course_recommendations(analysis)

            else:
                # Original tabs without JD analysis
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📋 Overview",
                    "📊 Skill Analysis",
                    "❌ Missing Skills",
                    "📚 Courses"
                ])

                with tab1:
                    display_basic_info(analysis)
                    st.markdown("---")
                    display_extracted_skills(analysis)
                    st.markdown("---")
                    display_category_prediction(analysis)

                with tab2:
                    display_skill_analysis(analysis)

                with tab3:
                    display_missing_skills(analysis)

                with tab4:
                    display_course_recommendations(analysis)

            # Download button
            st.markdown("---")
            download_data = analysis.copy()
            if stored_use_jd and stored_jd:
                download_data['job_description'] = {
                    'text': stored_jd,
                    'extracted_skills': stored_jd_skills,
                    'total_jd_skills': len(stored_jd_skills)
                }

            json_str = json.dumps(download_data, indent=2)
            st.download_button(
                label="📥 Download Full Report (JSON)",
                data=json_str,
                file_name=f"resume_analysis_{analysis['metadata']['analysis_date']}.json",
                mime="application/json",
                use_container_width=True
            )


if __name__ == '__main__':
    main()