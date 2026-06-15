"""Evaluate the MedDx pipeline on a golden set of clinical case vignettes.

Measures anti-bias invariants and hypothesis quality against known diagnoses.
Each case has a ground-truth diagnosis and a must-not-miss list.

Usage:
    python scripts/run_eval.py                         # all 10 cases
    python scripts/run_eval.py --cases 3               # first N cases
    python scripts/run_eval.py --out data/eval.json    # save JSON report
    make eval

Requires:
    - LLM API key in .env  (OPENAI_API_KEY or Ollama)
    - Qdrant running for Evidence agent  (evidence_symmetry / citation_count)
      without Qdrant those two metrics are 0 — all others still work

Exit code:
    0  mnm_recall == 100 %  (must-not-miss never skipped)
    1  at least one must-not-miss diagnosis was missed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from meddx.graph import build_graph
from meddx.schemas import (
    UNAVAILABLE,
    DiagnosticState,
    HypothesisEvidence,
    PatientCase,
    Phase,
)


# ---------------------------------------------------------------------------
# Golden case definition
# ---------------------------------------------------------------------------

@dataclass
class GoldenCase:
    case_id: str
    description: str
    bias_type: str          # which cognitive bias this case targets
    patient_case: PatientCase
    ground_truth: str       # correct diagnosis (lowercase)
    must_not_miss: list[str]  # dangerous diagnoses that must appear in hypotheses
    expected_top5: list[str]  # acceptable diagnoses for top-5 recall check


# ---------------------------------------------------------------------------
# 10 clinical vignettes — one per cognitive bias type, diverse organ systems
# ---------------------------------------------------------------------------

GOLDEN_CASES: list[GoldenCase] = [
    GoldenCase(
        case_id="case_01_stemi",
        description="Classic STEMI — anchoring risk to MI, must also flag dissection and PE",
        bias_type="anchoring",
        patient_case=PatientCase(
            chief_complaint="Crushing chest pain radiating to left arm, onset 40 minutes ago",
            history_of_present_illness=(
                "67-year-old male with sudden crushing substernal chest pain radiating to "
                "left arm and jaw, 9/10 intensity. Associated diaphoresis and nausea. "
                "No positional change. Symptoms started at rest."
            ),
            past_medical_history="Hypertension, type 2 diabetes mellitus, hyperlipidemia. CABG 5 years ago.",
            medications="Metformin 1000 mg bid, lisinopril 10 mg, atorvastatin 40 mg, aspirin 81 mg",
            family_history="Father died of MI at age 58. Mother: hypertension.",
            systems_review="Diaphoresis, nausea. No cough, no haemoptysis, no leg swelling or pain.",
            risk_factors="40 pack-year smoking history, obesity BMI 32, sedentary lifestyle, male sex",
            available_investigations="ECG: ST elevation V2–V4 and II, III, aVF. Troponin I pending. CXR: cardiomegaly, no widened mediastinum.",
        ),
        ground_truth="st-elevation myocardial infarction",
        must_not_miss=["myocardial infarction", "aortic dissection", "pulmonary embolism"],
        expected_top5=["myocardial infarction", "stemi", "acs", "acute coronary", "aortic dissection", "pulmonary embolism", "angina"],
    ),

    GoldenCase(
        case_id="case_02_sah",
        description="Thunderclap headache — anchoring to migraine; SAH is must-not-miss",
        bias_type="anchoring",
        patient_case=PatientCase(
            chief_complaint="Worst headache of my life, sudden onset during exercise",
            history_of_present_illness=(
                "38-year-old woman, sudden 'thunderclap' headache peaking within seconds "
                "during gym workout. 10/10 severity. Never had anything like this. "
                "Brief loss of consciousness, now alert but confused. Neck stiffness noted."
            ),
            past_medical_history="Occasional migraines (monthly, typical aura), no prior neurological events.",
            medications="Oral contraceptive pill, sumatriptan PRN",
            family_history="No family history of aneurysm or stroke.",
            systems_review="Neck stiffness, photophobia. No fever. No focal neurological deficits. Vomiting once.",
            risk_factors="OCP use, family history of 'bad headaches', moderate alcohol use",
            available_investigations="CT head (plain): no obvious bleed seen. LP: pending. BP 158/94.",
        ),
        ground_truth="subarachnoid haemorrhage",
        must_not_miss=["subarachnoid haemorrhage", "subarachnoid hemorrhage", "meningitis", "cerebral venous thrombosis"],
        expected_top5=["subarachnoid", "sah", "intracranial haemorrhage", "meningitis", "cerebral venous", "migraine"],
    ),

    GoldenCase(
        case_id="case_03_pe",
        description="PE in young woman on OCP — confirmation bias toward musculoskeletal",
        bias_type="confirmation_bias",
        patient_case=PatientCase(
            chief_complaint="Sudden shortness of breath and right-sided pleuritic chest pain, 2 days",
            history_of_present_illness=(
                "24-year-old woman, 3 weeks after long-haul flight (14 hours). "
                "Sudden onset dyspnoea and sharp right-sided chest pain worse on inspiration. "
                "Right calf swelling and tenderness since last week. SpO2 93 % on air."
            ),
            past_medical_history="No prior DVT or PE. No recent surgery.",
            medications="Combined oral contraceptive pill (3 years), no other regular medications",
            family_history="Mother had DVT post-partum.",
            systems_review="Right calf swelling, tenderness. Tachycardia. No fever. No haemoptysis.",
            risk_factors="OCP, prolonged immobility (long-haul flight), family history DVT, sedentary job",
            available_investigations="ECG: sinus tachycardia, S1Q3T3 pattern. D-dimer: 2.4 µg/mL (elevated). CXR: normal. HR 118 bpm.",
        ),
        ground_truth="pulmonary embolism",
        must_not_miss=["pulmonary embolism", "deep vein thrombosis"],
        expected_top5=["pulmonary embolism", "dvt", "pneumothorax", "pneumonia", "pleurisy", "myocarditis"],
    ),

    GoldenCase(
        case_id="case_04_meningitis",
        description="Bacterial meningitis — premature closure on viral syndrome",
        bias_type="premature_closure",
        patient_case=PatientCase(
            chief_complaint="Fever, severe headache, neck stiffness, 12 hours",
            history_of_present_illness=(
                "19-year-old university student, 12 hours of rapidly worsening headache, "
                "fever 39.8 °C, photophobia, and neck stiffness. Non-blanching petechial "
                "rash appeared on trunk 2 hours ago. Lives in dormitory, recent 'flu-like' "
                "illness in flatmate."
            ),
            past_medical_history="Healthy. Vaccinations: MenC at age 13 (not ACWY).",
            medications="No regular medications",
            family_history=UNAVAILABLE,
            systems_review="High fever, photophobia, phonophobia, neck stiffness. Non-blanching petechiae trunk/lower limbs. Kernig and Brudzinski signs positive.",
            risk_factors="University student living in dormitory, recent close contact with ill person, incomplete meningococcal vaccination",
            available_investigations="WBC 22 000 (neutrophilia). CRP 280 mg/L. Blood cultures taken. LP deferred pending CT.",
        ),
        ground_truth="bacterial meningitis",
        must_not_miss=["bacterial meningitis", "meningococcal", "meningitis", "septicaemia", "herpes encephalitis"],
        expected_top5=["bacterial meningitis", "meningococcal", "sepsis", "viral meningitis", "encephalitis", "meningitis"],
    ),

    GoldenCase(
        case_id="case_05_hcm",
        description="HCM in young athlete — availability bias toward vasovagal syncope",
        bias_type="availability_bias",
        patient_case=PatientCase(
            chief_complaint="Syncope during basketball training, 18-year-old male athlete",
            history_of_present_illness=(
                "18-year-old elite basketball player lost consciousness during intense "
                "training drill. No prodrome. Recovered spontaneously in 30 seconds. "
                "Third episode this season; previous two attributed to dehydration. "
                "Cousin died suddenly at age 22 during sport."
            ),
            past_medical_history="No known cardiac disease. No prior investigations.",
            medications="Creatine supplements, vitamin D",
            family_history="Cousin (male, 22 y/o) sudden cardiac death during football. Maternal uncle: 'heart problem'.",
            systems_review="Exertional syncope without prodrome. No palpitations before loss of consciousness. No seizure-like activity reported.",
            risk_factors="High-intensity competitive sport, family history of sudden cardiac death, male sex, adolescent age",
            available_investigations="ECG: left ventricular hypertrophy, deep Q waves in lateral leads. Echocardiogram: pending. BP 122/78.",
        ),
        ground_truth="hypertrophic cardiomyopathy",
        must_not_miss=["hypertrophic cardiomyopathy", "long qt syndrome", "arrhythmia", "wolff-parkinson-white", "channelopathy"],
        expected_top5=["hypertrophic cardiomyopathy", "hcm", "arrhythmia", "long qt", "wpw", "aortic stenosis", "cardiac"],
    ),

    GoldenCase(
        case_id="case_06_lymphoma",
        description="Hodgkin lymphoma — search satisficing on infectious mono explanation",
        bias_type="search_satisficing",
        patient_case=PatientCase(
            chief_complaint="6 weeks of fatigue, drenching night sweats, unexplained weight loss",
            history_of_present_illness=(
                "26-year-old woman, 6 weeks of progressive fatigue, drenching night sweats "
                "soaking the sheets, and unintentional 8 kg weight loss. Painless cervical "
                "lymphadenopathy noticed 4 weeks ago. No fever currently. Alcohol triggers "
                "pain in neck nodes (Hoster sign)."
            ),
            past_medical_history="Treated EBV mononucleosis 18 months ago. No other significant history.",
            medications="Combined OCP",
            family_history="No haematological malignancy.",
            systems_review="B symptoms: night sweats, weight loss > 10 % in 6 weeks, fever. Pruritus. Bilateral cervical and supraclavicular lymphadenopathy (firm, non-tender, rubbery). Splenomegaly on palpation.",
            risk_factors="Young adult female, prior EBV infection, female sex (Hodgkin's bimodal peak)",
            available_investigations="FBC: mild anaemia Hb 10.2, lymphocytosis. ESR 88 mm/h. LDH elevated. CXR: mediastinal widening. HIV negative.",
        ),
        ground_truth="hodgkin lymphoma",
        must_not_miss=["lymphoma", "leukaemia", "leukemia", "tuberculosis", "hiv"],
        expected_top5=["hodgkin", "lymphoma", "non-hodgkin", "leukaemia", "leukemia", "tb", "tuberculosis", "sarcoidosis"],
    ),

    GoldenCase(
        case_id="case_07_cholangitis",
        description="Acute cholangitis (Charcot's triad) — premature closure on hepatitis",
        bias_type="premature_closure",
        patient_case=PatientCase(
            chief_complaint="RUQ pain, jaundice, and fever since yesterday",
            history_of_present_illness=(
                "72-year-old woman, 24 hours of right upper quadrant pain, jaundice, "
                "and fever 38.9 °C (Charcot's triad). History of known cholelithiasis. "
                "Rigors. Mildly confused on assessment (Reynold's pentad). "
                "BP 92/60 mmHg, HR 112 bpm."
            ),
            past_medical_history="Known gallstones (asymptomatic until now). Type 2 diabetes. CKD stage 2.",
            medications="Metformin, amlodipine, atorvastatin",
            family_history=UNAVAILABLE,
            systems_review="RUQ pain, jaundice, fever with rigors, mild confusion. Murphy's sign positive. No diarrhoea.",
            risk_factors="Known cholelithiasis, female, age > 70, diabetes mellitus, prior ERCP",
            available_investigations="Bilirubin: 98 µmol/L. ALP: 420 U/L. ALT: 180 U/L. WBC: 19 500 (neutrophilia). Ultrasound: dilated CBD 12 mm, gallstones, no free fluid.",
        ),
        ground_truth="acute cholangitis",
        must_not_miss=["cholangitis", "sepsis", "cholecystitis", "hepatitis"],
        expected_top5=["cholangitis", "cholecystitis", "hepatitis", "pancreatitis", "sepsis", "biliary"],
    ),

    GoldenCase(
        case_id="case_08_lung_cancer",
        description="Lung cancer — availability bias toward COPD exacerbation in smoker",
        bias_type="availability_bias",
        patient_case=PatientCase(
            chief_complaint="3-month history of haemoptysis, weight loss, and worsening cough",
            history_of_present_illness=(
                "58-year-old male ex-smoker (45 pack-years, quit 2 years ago). "
                "3-month history of haemoptysis (blood-streaked sputum), unintentional "
                "7 kg weight loss, progressive exertional dyspnoea, and hoarseness. "
                "Recurrent 'chest infections' over 6 months."
            ),
            past_medical_history="COPD (GOLD stage II), hypertension, ex-smoker.",
            medications="Salbutamol inhaler PRN, tiotropium, ramipril",
            family_history="Father: lung cancer (deceased). No other malignancy.",
            systems_review="Haemoptysis, weight loss, hoarseness, dyspnoea. Finger clubbing noted. Supraclavicular lymphadenopathy on left. Reduced breath sounds left lower zone.",
            risk_factors="Heavy smoking history (45 pack-years), COPD, occupational asbestos exposure (construction work), age > 50, male",
            available_investigations="CXR: left hilar mass, left lower lobe collapse. CT chest: 4.5 cm left upper lobe mass, mediastinal lymphadenopathy. Sputum cytology: pending.",
        ),
        ground_truth="lung cancer",
        must_not_miss=["lung cancer", "malignancy", "tuberculosis", "pulmonary embolism"],
        expected_top5=["lung cancer", "carcinoma", "malignancy", "tuberculosis", "lymphoma", "mesothelioma", "copd"],
    ),

    GoldenCase(
        case_id="case_09_haemolytic_anaemia",
        description="Autoimmune haemolytic anaemia — confirmation bias toward iron deficiency",
        bias_type="confirmation_bias",
        patient_case=PatientCase(
            chief_complaint="Progressive fatigue, pallor, and jaundice over 3 weeks; dark urine",
            history_of_present_illness=(
                "45-year-old woman, 3-week history of rapidly worsening fatigue, "
                "pallor, and new onset jaundice. Dark (cola-coloured) urine noted. "
                "No blood loss. No travel. Started on a new medication (methyldopa) "
                "4 weeks ago for hypertension."
            ),
            past_medical_history="Hypertension (newly diagnosed 6 weeks ago). SLE diagnosed 2 years ago, currently quiescent.",
            medications="Methyldopa 250 mg tid (new, started 4 weeks ago), hydroxychloroquine",
            family_history="No haemolytic disorders. Mother: hypothyroidism.",
            systems_review="Fatigue, pallor, scleral icterus, splenomegaly. No lymphadenopathy. Dark urine, no haematuria on dipstick.",
            risk_factors="SLE (autoimmune predisposition), new methyldopa (known trigger), female sex, middle age",
            available_investigations="Hb 6.8 g/dL (normocytic). Reticulocytes 18 %. Bilirubin 62 µmol/L (indirect dominant). LDH elevated. Haptoglobin undetectable. DAT (Coombs): positive IgG. Blood film: spherocytes.",
        ),
        ground_truth="autoimmune haemolytic anaemia",
        must_not_miss=["haemolytic anaemia", "hemolytic anemia", "leukemia", "leukaemia", "g6pd"],
        expected_top5=["haemolytic", "hemolytic", "autoimmune", "drug-induced", "iron deficiency", "anaemia", "anemia"],
    ),

    GoldenCase(
        case_id="case_10_septic_arthritis",
        description="Septic arthritis — search satisficing on gout diagnosis",
        bias_type="search_satisficing",
        patient_case=PatientCase(
            chief_complaint="Hot, swollen, extremely painful right knee, unable to weight-bear, 2 days",
            history_of_present_illness=(
                "54-year-old male, 2-day history of acutely hot, swollen, erythematous "
                "right knee with severe pain (8/10) preventing weight bearing. "
                "Fever 38.6 °C. History of gout (1 prior attack in left great toe, 5 years ago). "
                "Recent dental procedure 2 weeks ago. No trauma."
            ),
            past_medical_history="Gout (one previous attack), hypertension, obesity.",
            medications="Amlodipine, allopurinol (started 3 months ago)",
            family_history="Father: gout.",
            systems_review="Fever, rigors. Hot erythematous right knee, grossly effused, restricted movement. No skin rash. No urethral discharge.",
            risk_factors="Recent bacteraemia source (dental procedure), gout history (confused with diagnosis), joint prosthesis absent, obesity, immunocompetent",
            available_investigations="WBC 18 200 (neutrophilia). CRP 210 mg/L. Uric acid: 0.52 mmol/L (elevated). Joint aspirate: 85 000 WBC/mm³, 95% neutrophils. Gram stain: Gram-positive cocci in clusters. Culture: pending.",
        ),
        ground_truth="septic arthritis",
        must_not_miss=["septic arthritis", "sepsis", "osteomyelitis"],
        expected_top5=["septic arthritis", "gout", "pseudogout", "reactive arthritis", "sepsis", "arthritis"],
    ),
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case_id: str
    description: str
    bias_type: str
    ground_truth: str

    phase_reached: str = "unknown"
    error: str | None = None

    # Hypothesis quality
    hyp_count: int = 0
    hyp_count_ok: bool = False      # >= 5 generated
    organ_systems: int = 0

    # Core recall
    top5_recall: bool = False
    top3_recall: bool = False

    # Anti-bias invariant
    mnm_recall: bool = False
    mnm_missing: list[str] = field(default_factory=list)

    # Evidence quality (requires Qdrant corpus)
    evidence_symmetry: float = 0.0  # fraction of hypotheses with both FOR + AGAINST
    citation_count: int = 0

    elapsed_s: float = 0.0


def _term_in_name(name: str, terms: list[str]) -> bool:
    """True if any term is a substring of the hypothesis name (case-insensitive)."""
    n = name.lower()
    return any(t in n for t in terms)


def _compute(case: GoldenCase, result: dict, elapsed: float) -> CaseResult:
    hypotheses = result.get("hypotheses") or []
    evidence: list[HypothesisEvidence] = result.get("evidence") or []

    hyp_names = [h.name for h in hypotheses]

    top5_recall = any(_term_in_name(n, case.expected_top5) for n in hyp_names[:5])
    top3_recall = any(_term_in_name(n, case.expected_top5) for n in hyp_names[:3])

    mnm_missing = [
        m for m in case.must_not_miss
        if not any(_term_in_name(n, [m]) for n in hyp_names)
    ]

    ev_with_both = sum(1 for e in evidence if e.supporting and e.refuting)
    ev_symmetry = ev_with_both / len(evidence) if evidence else 0.0
    citations = sum(len(e.supporting) + len(e.refuting) for e in evidence)

    return CaseResult(
        case_id=case.case_id,
        description=case.description,
        bias_type=case.bias_type,
        ground_truth=case.ground_truth,
        phase_reached=str(result.get("phase", "unknown")),
        hyp_count=len(hypotheses),
        hyp_count_ok=len(hypotheses) >= 5,
        organ_systems=len({h.organ_system for h in hypotheses}),
        top5_recall=top5_recall,
        top3_recall=top3_recall,
        mnm_recall=len(mnm_missing) == 0,
        mnm_missing=mnm_missing,
        evidence_symmetry=ev_symmetry,
        citation_count=citations,
        elapsed_s=round(elapsed, 1),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_case(graph, case: GoldenCase) -> CaseResult:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    t0 = time.time()
    try:
        result = graph.invoke(
            DiagnosticState(patient_case=case.patient_case),
            config=config,
        )

        # Intake gate triggered — case data is incomplete (should not happen for golden cases)
        if result.get("phase") == Phase.AWAITING_DATA:
            missing = result.get("missing_fields", [])
            return CaseResult(
                case_id=case.case_id, description=case.description,
                bias_type=case.bias_type, ground_truth=case.ground_truth,
                error=f"Intake gate blocked — missing fields: {missing}",
                elapsed_s=round(time.time() - t0, 1),
            )

        # Synthesis interrupt — auto-submit a neutral student ranking so the
        # pipeline completes and we can measure all downstream metrics.
        if result.get("phase") == Phase.SYNTHESIS and result.get("synthesis") is None:
            hyps = result.get("hypotheses") or []
            neutral_ranking = ", ".join(str(i + 1) for i in range(len(hyps)))
            result = graph.invoke(Command(resume=neutral_ranking), config=config)

        return _compute(case, result, time.time() - t0)

    except Exception as exc:
        return CaseResult(
            case_id=case.case_id, description=case.description,
            bias_type=case.bias_type, ground_truth=case.ground_truth,
            error=str(exc),
            elapsed_s=round(time.time() - t0, 1),
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_TICK = "✓"
_CROSS = "✗"

def _bool(v: bool) -> str:
    return _TICK if v else _CROSS


def _print_report(results: list[CaseResult]) -> None:
    ok = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    print("\n" + "═" * 80)
    print("  MedDx Evaluation Report")
    print("═" * 80)

    # Per-case table
    header = f"{'ID':<22} {'bias':<20} {'hyp≥5':<6} {'top3':<5} {'top5':<5} {'MNM':<5} {'ev_sym':<7} {'cit':<5} {'t(s)'}"
    print(f"\n{header}")
    print("─" * 80)
    for r in results:
        if r.error:
            print(f"{r.case_id:<22} {'ERROR':<20} — {r.error[:45]}")
            continue
        print(
            f"{r.case_id:<22} {r.bias_type:<20} "
            f"{_bool(r.hyp_count_ok):<6} {_bool(r.top3_recall):<5} "
            f"{_bool(r.top5_recall):<5} {_bool(r.mnm_recall):<5} "
            f"{r.evidence_symmetry:>5.0%}  {r.citation_count:>4}  {r.elapsed_s}"
        )
        if r.mnm_missing:
            print(f"  {'':20}  ↳ missing must-not-miss: {', '.join(r.mnm_missing)}")

    # Aggregate
    n = len(ok)
    if n == 0:
        print("\nAll cases failed — check LLM configuration.")
        return

    mnm_all  = sum(r.mnm_recall for r in ok)
    top5_all = sum(r.top5_recall for r in ok)
    top3_all = sum(r.top3_recall for r in ok)
    hyp5_all = sum(r.hyp_count_ok for r in ok)
    avg_sym  = sum(r.evidence_symmetry for r in ok) / n
    avg_cit  = sum(r.citation_count for r in ok) / n
    avg_org  = sum(r.organ_systems for r in ok) / n

    print("\n" + "─" * 80)
    print(f"  Ran {n} cases  ({len(failed)} errors)")
    print()
    print(f"  {'must-not-miss recall':<30}  {mnm_all}/{n}  {'← must be 100 %' if mnm_all < n else '← ✓ PASS'}")
    print(f"  {'top-5 recall':<30}  {top5_all}/{n}")
    print(f"  {'top-3 recall':<30}  {top3_all}/{n}")
    print(f"  {'hypotheses ≥ 5 (invariant)':<30}  {hyp5_all}/{n}")
    print(f"  {'avg organ systems covered':<30}  {avg_org:.1f}")
    print(f"  {'avg evidence symmetry':<30}  {avg_sym:.0%}  {'(0% = corpus empty)' if avg_sym == 0 else ''}")
    print(f"  {'avg citations per case':<30}  {avg_cit:.1f}")
    print("═" * 80)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MedDx golden-set evaluation")
    parser.add_argument(
        "--cases", type=int, default=None,
        help="Run only the first N cases (default: all 10)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Path to save JSON report (e.g. data/eval.json)",
    )
    args = parser.parse_args(argv)

    cases = GOLDEN_CASES[: args.cases] if args.cases else GOLDEN_CASES
    print(f"Building graph …  ({len(cases)} cases)")
    graph = build_graph(checkpointer=MemorySaver())

    results: list[CaseResult] = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.case_id} …", end=" ", flush=True)
        r = _run_case(graph, case)
        tag = "✓" if r.error is None and r.mnm_recall else ("✗ MNM!" if not r.mnm_recall else "ERR")
        print(tag)
        results.append(r)

    _print_report(results)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
        print(f"\nJSON report saved → {out_path}")

    # Exit 1 if any completed case missed a must-not-miss diagnosis
    any_mnm_miss = any(not r.mnm_recall for r in results if r.error is None)
    return 1 if any_mnm_miss else 0


if __name__ == "__main__":
    sys.exit(main())
