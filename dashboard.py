"""Live trash-detection dashboard — run with: streamlit run dashboard.py"""
from __future__ import annotations

import json
import time
import datetime
from collections import deque, Counter
from pathlib import Path

import numpy as np
import streamlit as st

JSONL_PATH = "detections.jsonl"
REFRESH_S = 2
MAX_RECORDS = 200
COLS = 3

MATERIAL_COLOURS = {
    "plastic": "#00C8FF",
    "paper":   "#32DC32",
    "metal":   "#B464FF",
    "glass":   "#FFC832",
    "organic": "#32B464",
    "other":   "#B4B4B4",
}


def load_tail(path: str, n: int) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    buf = deque(maxlen=n)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                buf.append(line)
    records = []
    for line in reversed(buf):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def sidebar_stats(records: list[dict]) -> None:
    with st.sidebar:
        st.title("Stats")
        st.metric("Session start", st.session_state["t0"].strftime("%H:%M:%S"))
        st.metric("Recent records", len(records))
        st.divider()
        counts = Counter(r.get("material") or "unknown" for r in records)
        for mat in ["plastic", "paper", "metal", "glass", "organic", "other", "unknown"]:
            if counts[mat]:
                colour = MATERIAL_COLOURS.get(mat, "#B4B4B4")
                st.markdown(
                    f'<span style="color:{colour}">■</span> **{mat.capitalize()}**: {counts[mat]}',
                    unsafe_allow_html=True,
                )


def detection_card(record: dict) -> None:
    snap = record.get("snapshot_path")
    material = record.get("material") or "unknown"
    conf = record.get("material_confidence") or record.get("confidence") or 0.0
    ts = record.get("timestamp", "")[:19].replace("T", " ")

    with st.container(border=True):
        if snap and Path(snap).exists():
            st.image(snap, use_container_width=True)
        else:
            placeholder = np.full((80, 120, 3), 70, dtype=np.uint8)
            st.image(placeholder, use_container_width=True)

        colour = MATERIAL_COLOURS.get(material, "#B4B4B4")
        st.markdown(
            f'<b style="color:{colour}">{material.upper()}</b>',
            unsafe_allow_html=True,
        )
        st.caption(f"{conf:.0%} · {ts} UTC")


def main() -> None:
    st.set_page_config(layout="wide", page_title="Trash Detection", page_icon="♻️")

    if "t0" not in st.session_state:
        st.session_state["t0"] = datetime.datetime.now()

    records = load_tail(JSONL_PATH, MAX_RECORDS)
    sidebar_stats(records)

    st.title("Live Trash Detection Feed")

    if not records:
        st.info("No detections yet — start `detect.py` in another terminal.")
    else:
        snap_records = [r for r in records if r.get("snapshot_path")]
        display = snap_records if snap_records else records
        cols = st.columns(COLS)
        for i, rec in enumerate(display):
            with cols[i % COLS]:
                detection_card(rec)

    time.sleep(REFRESH_S)
    st.rerun()


if __name__ == "__main__":
    main()
