"""Streamlit UI entry point (stub).

Run: streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="MedDx — диференційна діагностика", page_icon="🩺")

st.title("🩺 MedDx — навчальна диференційна діагностика")
st.warning(
    "Освітній інструмент для студентів-медиків. "
    "Не призначений для клінічного застосування і не замінює лікаря."
)

st.info("Каркас проєкту. Діагностичний пайплайн (LangGraph) буде підключено тут.")
