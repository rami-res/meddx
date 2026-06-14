"""Streamlit UI for MedDx — multi-agent RAG diagnostic pipeline.

Educational tool only — not for clinical use.

Session flow:
  1. Patient case form → graph.invoke(DiagnosticState)
  2. [AWAITING_DATA loop] — LLM intake question + update form for missing fields
  3. Pipeline runs automatically: Hypothesis → Evidence → Devil's Advocate →
     Root Cause → Synthesis (hits interrupt)
  4. SYNTHESIS interrupt — student ranks hypotheses → Command(resume=ranking)
  5. DONE — ranked differential + workup plan + Socratic feedback + citations
"""

from __future__ import annotations

import uuid

import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from meddx.graph import build_graph
from meddx.schemas import (
    UNAVAILABLE,
    Citation,
    DiagnosticState,
    Hypothesis,
    HypothesisEvidence,
    PatientCase,
    Phase,
    RankedHypothesis,
    SynthesisResult,
)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MedDx — диференційна діагностика",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached resources — survive Streamlit re-runs in the same process
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_graph():
    checkpointer = MemorySaver()
    return build_graph(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_session() -> None:
    defaults = {
        "thread_id": str(uuid.uuid4()),
        "result": None,          # last dict returned by graph.invoke()
        "case_values": {},       # raw form values survive AWAITING_DATA loops
        "error": None,           # graph-level error message
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _config() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def _reset() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.result = None
    st.session_state.case_values = {}
    st.session_state.error = None


# ---------------------------------------------------------------------------
# Graph invocation helpers
# ---------------------------------------------------------------------------

def _invoke(patient_case: PatientCase) -> None:
    graph = _get_graph()
    st.session_state.error = None
    try:
        result = graph.invoke(
            DiagnosticState(patient_case=patient_case),
            config=_config(),
        )
        st.session_state.result = result
    except Exception as exc:
        st.session_state.error = str(exc)
    st.rerun()


def _resume(student_ranking: str) -> None:
    graph = _get_graph()
    st.session_state.error = None
    try:
        result = graph.invoke(
            Command(resume=student_ranking),
            config=_config(),
        )
        st.session_state.result = result
    except Exception as exc:
        st.session_state.error = str(exc)
    st.rerun()


# ---------------------------------------------------------------------------
# Form field definitions
# ---------------------------------------------------------------------------

_FIELD_DEFS: list[tuple[str, str, bool]] = [
    # (field_name, label, multiline)
    ("chief_complaint",              "Основна скарга",                    False),
    ("history_of_present_illness",   "Анамнез хвороби",                   True),
    ("past_medical_history",         "Минула медична історія",            True),
    ("medications",                  "Поточні медикаменти",               True),
    ("family_history",               "Сімейний анамнез",                  False),
    ("systems_review",               "Огляд систем органів",              True),
    ("risk_factors",                 "Фактори ризику",                    False),
    ("available_investigations",     "Наявні дослідження / аналізи",     True),
]

_NA_SYNONYMS = {"not available", "unavailable", "недоступно", "н/д", "n/a", "na", "—", "-"}


def _normalize_field(raw: str) -> str | None:
    stripped = raw.strip()
    if not stripped:
        return None
    if stripped.lower() in _NA_SYNONYMS:
        return UNAVAILABLE
    return stripped


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar(result: dict | None) -> None:
    with st.sidebar:
        st.markdown("## 🩺 MedDx")
        st.caption("Навчальна система для студентів-медиків")

        st.warning(
            "**Лише для навчання.**  \n"
            "Не призначений для клінічного застосування і не замінює лікаря.",
            icon="⚠️",
        )

        st.divider()

        if result is not None:
            phase = result.get("phase")
            _PHASE_LABELS = {
                Phase.INTAKE:        ("⏳", "Збір даних"),
                Phase.AWAITING_DATA: ("📋", "Потрібно більше даних"),
                Phase.HYPOTHESES:    ("🧠", "Генерація гіпотез"),
                Phase.EVIDENCE:      ("🔬", "Пошук доказів"),
                Phase.CHALLENGE:     ("👿", "Адвокат диявола"),
                Phase.ROOT_CAUSE:    ("🌳", "Першопричина"),
                Phase.SYNTHESIS:     ("⏸️", "Очікування відповіді"),
                Phase.DONE:          ("✅", "Аналіз завершено"),
            }
            icon, label = _PHASE_LABELS.get(phase, ("❓", str(phase)))
            st.markdown(f"**Статус:** {icon} {label}")

            hyps = result.get("hypotheses") or []
            evs  = result.get("evidence")   or []
            if hyps:
                st.markdown(f"Гіпотез: **{len(hyps)}**")
            if evs:
                total_cit = sum(len(e.supporting) + len(e.refuting) for e in evs)
                st.markdown(f"Цитат знайдено: **{total_cit}**")

            st.divider()

        if st.button("🔄 Новий кейс", use_container_width=True):
            _reset()
            st.rerun()

        st.divider()
        st.caption(
            "Джерела: Europe PMC · PubMed · PMC · DOAJ · BMC · PLOS · Cureus  \n"
            "Ранжування за рівнем доказовості, а не протоколами однієї країни."
        )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_STUDY_TYPE_LABEL: dict[str, str] = {
    "meta-analysis":      "🥇 Meta-analysis (L1)",
    "systematic-review":  "🥈 Systematic review (L2)",
    "rct":                "🥉 RCT (L3)",
    "cohort":             "📊 Cohort (L4)",
    "case-control":       "🔍 Case-control (L5)",
    "case-report":        "📄 Case report (L6)",
    "review":             "📚 Review (L7)",
    "other":              "📝 Other",
}


def _study_badge(study_type: str) -> str:
    return _STUDY_TYPE_LABEL.get(study_type.lower(), f"📝 {study_type}")


def _citation_line(c: Citation) -> str:
    parts = [f"**{c.title}**", f"*{c.journal}*", str(c.year), _study_badge(c.study_type)]
    if c.doi:
        parts.append(f"[DOI:{c.doi}](https://doi.org/{c.doi})")
    elif c.pmid:
        parts.append(f"[PMID:{c.pmid}](https://pubmed.ncbi.nlm.nih.gov/{c.pmid}/)")
    return " — ".join(parts)


def _render_hypotheses_tab(result: dict) -> None:
    hypotheses: list[Hypothesis] = result.get("hypotheses") or []
    if not hypotheses:
        st.info("Гіпотези ще не згенеровані.")
        return

    st.markdown(f"Агент згенерував **{len(hypotheses)}** гіпотез (не ранжованих).")
    for h in hypotheses:
        badge = "🚨 **must-not-miss**" if h.is_must_not_miss else ""
        with st.expander(f"{h.name} — *{h.organ_system}* {badge}"):
            st.markdown(h.rationale)


def _render_evidence_tab(result: dict) -> None:
    evidence: list[HypothesisEvidence] = result.get("evidence") or []
    hypotheses: list[Hypothesis]       = result.get("hypotheses") or []

    if not evidence:
        st.info("Докази ще не отримані.")
        return

    name_by_id = {h.id: h.name for h in hypotheses}
    for ev in evidence:
        hyp_name = name_by_id.get(ev.hypothesis_id, ev.hypothesis_id)
        total = len(ev.supporting) + len(ev.refuting)
        with st.expander(f"**{hyp_name}** — {len(ev.supporting)} ✅ підтверджуючих · {len(ev.refuting)} ❌ спростовуючих ({total} всього)"):
            if ev.supporting:
                st.markdown("**Підтверджуючі:**")
                for c in ev.supporting:
                    st.markdown(f"- {_citation_line(c)}")
            if ev.refuting:
                st.markdown("**Спростовуючі:**")
                for c in ev.refuting:
                    st.markdown(f"- {_citation_line(c)}")
            if not ev.supporting and not ev.refuting:
                st.caption("Корпус порожній або Qdrant недоступний — докази не знайдені.")


def _render_challenge_tab(result: dict) -> None:
    from meddx.schemas import ChallengeReport
    challenge: ChallengeReport | None = result.get("challenge")

    if not challenge:
        st.info("Адвокат диявола ще не запустився.")
        return

    st.markdown("### Суперечності")
    if challenge.contradictions:
        for c in challenge.contradictions:
            st.markdown(f"- {c}")
    else:
        st.caption("Суперечності не виявлено.")

    if challenge.alternative_explanations:
        st.markdown("### Альтернативні пояснення симптомів")
        for symptom, alt in challenge.alternative_explanations.items():
            st.markdown(f"- **{symptom}:** {alt}")

    if challenge.discriminating_test:
        st.success(f"**Ключове дослідження:** {challenge.discriminating_test}")


def _render_root_cause_tab(result: dict) -> None:
    from meddx.schemas import RootCauseAssessment
    rc: RootCauseAssessment | None = result.get("root_cause")

    if not rc:
        st.info("Аналіз першопричини ще не завершено.")
        return

    explained = rc.all_findings_explained
    if explained:
        st.success("✅ Всі знахідки пояснені поточним диференційним рядом.")
    else:
        st.warning("⚠️ Деякі знахідки залишаються незрозумілими — можлива системна причина.")

    if rc.unexplained_findings:
        st.markdown("**Непояснені знахідки:**")
        for f in rc.unexplained_findings:
            st.markdown(f"- {f}")

    if rc.candidate_underlying_conditions:
        st.markdown("**Кандидати на системний стан:**")
        for cond in rc.candidate_underlying_conditions:
            st.markdown(f"- {cond}")

    if rc.comment:
        st.caption(rc.comment)


def _render_synthesis_tab(result: dict) -> None:
    synthesis: SynthesisResult | None = result.get("synthesis")
    hypotheses: list[Hypothesis]      = result.get("hypotheses") or []

    if not synthesis:
        st.info("Синтез ще не завершено.")
        return

    name_by_id    = {h.id: h.name for h in hypotheses}
    is_mnm_by_id  = {h.id: h.is_must_not_miss for h in hypotheses}
    sorted_ranking = sorted(synthesis.ranking, key=lambda r: r.rank)

    # Ranked differential
    st.markdown("### Диференційний ряд")
    for r in sorted_ranking:
        name   = name_by_id.get(r.hypothesis_id, r.hypothesis_id)
        mnm    = " 🚨" if is_mnm_by_id.get(r.hypothesis_id) else ""
        badge  = {1: "🥇", 2: "🥈", 3: "🥉"}.get(r.rank, f"**#{r.rank}**")
        header = f"{badge} {name}{mnm}"

        with st.expander(header, expanded=r.rank <= 2):
            if r.probability_note:
                st.markdown(f"*{r.probability_note}*")
            if r.citations:
                st.markdown("**Ключові джерела:**")
                for c in r.citations[:3]:
                    st.markdown(f"- {_citation_line(c)}")
                if len(r.citations) > 3:
                    st.caption(f"... та ще {len(r.citations) - 3} цитат")

    # Evidence summary
    if synthesis.evidence_summary:
        st.markdown("### Якість доказової бази")
        st.info(synthesis.evidence_summary)

    # Workup plan
    if synthesis.workup_plan:
        st.markdown("### План подальшого обстеження")
        for i, step in enumerate(synthesis.workup_plan, 1):
            st.markdown(f"{i}. {step}")

    # Socratic feedback
    if synthesis.socratic_feedback:
        st.markdown("### Зворотній зв'язок по вашому ранжуванню")
        st.success(synthesis.socratic_feedback)


# ---------------------------------------------------------------------------
# Patient case form (shared between initial + AWAITING_DATA update)
# ---------------------------------------------------------------------------

def _render_patient_form(
    prefilled: dict,
    missing: list[str],
    intake_message: str | None,
) -> None:
    if intake_message:
        st.info(f"**Асистент:** {intake_message}")

    with st.form("patient_case_form", clear_on_submit=False):
        cols = st.columns(2)
        values: dict[str, str] = {}

        for idx, (field_name, label, multiline) in enumerate(_FIELD_DEFS):
            col = cols[idx % 2]
            current = prefilled.get(field_name) or ""
            if current == UNAVAILABLE:
                current = ""
            is_missing = field_name in missing
            display_label = f":red[{label}] ❗" if is_missing else label
            help_text = "Введіть «недоступно» якщо дані недоступні"

            with col:
                if multiline:
                    values[field_name] = st.text_area(
                        display_label, value=current, height=90, help=help_text, key=f"field_{field_name}"
                    )
                else:
                    values[field_name] = st.text_input(
                        display_label, value=current, help=help_text, key=f"field_{field_name}"
                    )

        st.divider()
        label = "Запустити діагностичний пайплайн" if not missing else "Оновити та продовжити"
        submitted = st.form_submit_button(label, type="primary", use_container_width=True)

    if submitted:
        normalized: dict[str, str | None] = {
            k: _normalize_field(v) for k, v in values.items()
        }
        st.session_state.case_values = {k: v for k, v in normalized.items() if v}
        with st.spinner("Запуск діагностичного пайплайну..."):
            _invoke(PatientCase(**normalized))


# ---------------------------------------------------------------------------
# Ranking form (Synthesis interrupt)
# ---------------------------------------------------------------------------

def _render_ranking_form(result: dict) -> None:
    hypotheses: list[Hypothesis] = result.get("hypotheses") or []

    st.markdown("---")
    st.markdown("## Ваша черга 🎓")
    st.markdown(
        "Перш ніж побачити підсумок системи, **ранжуйте гіпотези** "
        "від найбільш до найменш імовірних на основі кейсу."
    )

    # Numbered list for reference
    hyp_text = "\n".join(
        f"{i+1}. {h.name}" for i, h in enumerate(hypotheses)
    )
    st.code(hyp_text, language=None)

    with st.form("ranking_form"):
        ranking_input = st.text_area(
            "Ваше ранжування",
            placeholder='Наприклад: "1, 3, 2, 5, 4" або назви по порядку...',
            height=100,
            help="Введіть номери або назви гіпотез у порядку від найймовірнішої до найменш імовірної.",
        )
        submitted = st.form_submit_button("Відправити ранжування", type="primary", use_container_width=True)

    if submitted:
        if not ranking_input.strip():
            st.warning("Будь ласка, введіть ваше ранжування.")
        else:
            with st.spinner("Агент синтезу аналізує ваш вибір..."):
                _resume(ranking_input.strip())


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def _page_welcome() -> None:
    st.markdown(
        """
        ## Ласкаво просимо до MedDx

        MedDx — навчальна мульти-агентна RAG-система, що допомагає студентам-медикам
        формувати диференційний діагностичний ряд і протидіє когнітивним упередженням:

        | Упередження | Механізм протидії |
        |---|---|
        | Anchoring / Availability | Hypothesis-агент генерує ≥5 гіпотез з різних систем органів |
        | Confirmation bias | Evidence-агент шукає докази ЗА і ПРОТИ кожної гіпотези окремо |
        | Premature closure | Completeness gate блокує пайплайн до заповнення всіх полів |
        | Search satisficing | Root-Cause агент перевіряє, чи пояснює діагноз УСІ знахідки |

        **Заповніть форму пацієнта нижче, щоб розпочати.**
        """
    )
    st.divider()


def _page_patient_form_initial() -> None:
    _page_welcome()
    st.markdown("### Кейс пацієнта")
    _render_patient_form(
        prefilled=st.session_state.case_values,
        missing=[],
        intake_message=None,
    )


def _page_awaiting_data(result: dict) -> None:
    st.markdown("## Необхідна додаткова інформація")
    missing: list[str] = result.get("missing_fields") or []
    intake_message: str | None = result.get("intake_message")
    _render_patient_form(
        prefilled=st.session_state.case_values,
        missing=missing,
        intake_message=intake_message,
    )


def _page_pipeline_result(result: dict, interrupted_at_synthesis: bool) -> None:
    phase = result.get("phase")

    if interrupted_at_synthesis:
        st.markdown("## Пайплайн завершив попередній аналіз")
    else:
        st.markdown("## Результати діагностичного аналізу")

    # Build tab list dynamically based on what's available
    tab_labels = ["🧠 Гіпотези", "🔬 Докази", "👿 Адвокат диявола", "🌳 Першопричина"]
    if not interrupted_at_synthesis:
        tab_labels.append("🏆 Фінальний аналіз")

    tabs = st.tabs(tab_labels)

    with tabs[0]:
        _render_hypotheses_tab(result)
    with tabs[1]:
        _render_evidence_tab(result)
    with tabs[2]:
        _render_challenge_tab(result)
    with tabs[3]:
        _render_root_cause_tab(result)
    if not interrupted_at_synthesis and len(tabs) > 4:
        with tabs[4]:
            _render_synthesis_tab(result)

    if interrupted_at_synthesis:
        _render_ranking_form(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session()
    _render_sidebar(st.session_state.result)

    # Show any pipeline errors
    if st.session_state.error:
        st.error(f"Помилка пайплайну: {st.session_state.error}")

    result: dict | None = st.session_state.result

    if result is None:
        _page_patient_form_initial()
        return

    phase = result.get("phase")

    if phase == Phase.AWAITING_DATA:
        _page_awaiting_data(result)
        return

    # Detect synthesis interrupt: phase is SYNTHESIS but synthesis output is not yet set
    interrupted = (phase == Phase.SYNTHESIS and result.get("synthesis") is None)

    if interrupted or phase == Phase.DONE:
        _page_pipeline_result(result, interrupted_at_synthesis=interrupted)
        return

    # Unexpected phase — show raw state for debugging
    st.warning(f"Неочікувана фаза: {phase}")
    st.json({k: str(v) for k, v in result.items()})


main()
