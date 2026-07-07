from app.requirements import RequirementsExtractor


def test_extracts_role_and_strips_filler_words():
    req = RequirementsExtractor().extract(["Hiring a Java developer who works with stakeholders"])
    assert req.role == "java developer"
    assert "java" in req.skills
    assert req.personality_required is True
    assert req.seniority is None


def test_extracts_seniority_across_turns():
    req = RequirementsExtractor().extract([
        "Hiring a Java developer who works with stakeholders",
        "Mid-level, around 4 years",
    ])
    assert req.seniority == "mid-professional"


def test_extracts_multiple_skills_and_leadership():
    req = RequirementsExtractor().extract([
        "We need a senior data analyst with Python and Excel skills who can lead a small team"
    ])
    assert req.seniority == "senior"
    assert set(req.skills) == {"python", "excel"}
    assert req.leadership_required is True


def test_empty_message_yields_empty_requirements():
    req = RequirementsExtractor().extract(["I need an assessment"])
    assert req.is_effectively_empty()


def test_query_text_is_stable_and_nonempty_when_requirements_present():
    req = RequirementsExtractor().extract(["Hiring a mid-level Java developer"])
    text = req.as_query_text()
    assert "java developer" in text
    assert "mid-professional" in text or "mid" in text
