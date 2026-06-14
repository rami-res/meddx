"""Intake completeness gate (anti premature closure)."""

from meddx.schemas import UNAVAILABLE, PatientCase


def test_empty_case_reports_all_fields_missing():
    case = PatientCase()
    assert len(case.missing_fields()) == len(PatientCase.model_fields)


def test_unavailable_marker_counts_as_collected():
    case = PatientCase(
        chief_complaint="Chest pain",
        history_of_present_illness="2 days, episodic",
        past_medical_history="Hypertension",
        medications="Amlodipine",
        family_history=UNAVAILABLE,
        systems_review="Unremarkable",
        risk_factors="Smoker",
        available_investigations=UNAVAILABLE,
    )
    assert case.missing_fields() == []


def test_blank_string_is_still_missing():
    case = PatientCase(chief_complaint="   ")
    assert "chief_complaint" in case.missing_fields()
