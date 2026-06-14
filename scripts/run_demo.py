"""End-to-end demo of the diagnostic graph with stub agents.

Run: python scripts/run_demo.py
Shows both paths: an incomplete case stopped by the intake gate, and a
complete case flowing through all six agents to synthesis.
"""

from meddx.graph import build_graph
from meddx.schemas import UNAVAILABLE, DiagnosticState, PatientCase


def main() -> None:
    graph = build_graph()

    print("=" * 70)
    print("1) Incomplete case -> intake gate blocks the pipeline")
    print("=" * 70)
    incomplete = DiagnosticState(
        patient_case=PatientCase(chief_complaint="Chest pain for 2 days")
    )
    result = graph.invoke(incomplete)
    print(f"phase: {result['phase'].value}")
    print(f"missing fields: {', '.join(result['missing_fields'])}")
    print(f"hypotheses generated: {len(result['hypotheses'])} (expected 0)")

    print()
    print("=" * 70)
    print("2) Complete case -> full pipeline to synthesis")
    print("=" * 70)
    complete = DiagnosticState(
        patient_case=PatientCase(
            chief_complaint="Chest pain for 2 days",
            history_of_present_illness="Retrosternal, episodic, worse after meals",
            past_medical_history="Hypertension",
            medications="Amlodipine 5 mg",
            family_history="Father: MI at 56",
            systems_review="Intermittent low-grade fever; otherwise unremarkable",
            risk_factors="Smoker, 20 pack-years",
            available_investigations=UNAVAILABLE,
        )
    )
    result = graph.invoke(complete)

    print(f"phase: {result['phase'].value}")
    print(f"\nhypotheses ({len(result['hypotheses'])}, unranked):")
    for h in result["hypotheses"]:
        flag = " [MUST-NOT-MISS]" if h.is_must_not_miss else ""
        print(f"  - {h.name} ({h.organ_system}){flag}")

    print(f"\nevidence bundles: {len(result['evidence'])} (supporting+refuting each)")
    print(f"devil's advocate discriminating test: {result['challenge'].discriminating_test}")
    print(f"root cause — unexplained findings: {result['root_cause'].unexplained_findings}")

    print("\nfinal ranking (synthesis):")
    names = {h.id: h.name for h in result["hypotheses"]}
    for r in result["synthesis"].ranking:
        cites = ", ".join(c.pmid or c.doi or "?" for c in r.citations)
        print(f"  {r.rank}. {names[r.hypothesis_id]}  [citations: {cites}]")


if __name__ == "__main__":
    main()
