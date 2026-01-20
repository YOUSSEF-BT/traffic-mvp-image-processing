import time
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Traffic Dashboard", layout="wide")
st.title("Traffic MVP — Dashboard")

csv_path = st.sidebar.text_input("CSV path", "metrics.csv")
refresh = st.sidebar.slider("Refresh (s)", 1, 10, 2)

placeholder = st.empty()

while True:
    with placeholder.container():
        st.caption("Le dashboard lit metrics.csv (généré par traffic_mvp.py)")

        try:
            df = pd.read_csv(csv_path)

            # Trier par colonne temps si elle existe
            if "t_video_s" in df.columns:
                df = df.sort_values("t_video_s")
            elif "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
                df = df.sort_values("ts")

            st.subheader("Dernières mesures")
            st.dataframe(df.tail(20), use_container_width=True)

            st.subheader("Courbes")

            # Tracer en utilisant la bonne colonne comme index
            if "t_video_s" in df.columns:
                cols = [c for c in ["vehicles_in_frame", "count_total", "congestion_score"] if c in df.columns]
                if cols:
                    st.line_chart(df.set_index("t_video_s")[cols])
                else:
                    st.info("Colonnes à tracer introuvables (vehicles_in_frame / count_total / congestion_score).")

            elif "ts" in df.columns:
                cols = [c for c in ["vehicles_in_frame", "count_total", "congestion_score"] if c in df.columns]
                if cols:
                    st.line_chart(df.set_index("ts")[cols])
                else:
                    st.info("Colonnes à tracer introuvables (vehicles_in_frame / count_total / congestion_score).")

            else:
                st.info("Impossible de tracer : il manque la colonne t_video_s (ou ts).")

        except Exception as e:
            st.warning(f"En attente du fichier CSV... ({e})")

    time.sleep(refresh)
    st.rerun()
