import requests
import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

import streamlit as st

API = "http://127.0.0.1:8010"
session = requests.Session()
session.trust_env = False  # игнорировать системные HTTP_PROXY/HTTPS_PROXY


st.set_page_config(page_title="Backpack Finder", layout="wide")
st.title("🎒 Backpack Finder (YOLO + Mask R-CNN + Crop Classification)")

tabs = st.tabs(["Детекция", "Сегментация", "История", "Статистика/Отчёты"])

with tabs[0]:
    st.subheader("Загрузка изображения и детекция")
    st.caption("Классификация кропов (ImageNet): crop_top1_label/crop_top1_conf + прикладной bag_type (backpack/suitcase/handbag/bag/other).")
    col1, col2 = st.columns([1, 1])

    with col1:
        file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"], key="det")
        conf = st.slider("Confidence", min_value=0.05, max_value=0.95, value=0.25, step=0.05, key="det_conf")
        only_backpack = st.checkbox("Показывать только backpack", value=False, key="det_bp")
        run = st.button("Обработать", key="det_run")

    if run and file is not None:
        files = {"file": (file.name, file.getvalue(), file.type)}
        params = {"conf": conf, "only_backpack": only_backpack}
        r = session.post(f"{API}/infer/image", files=files, params=params, timeout=180)
        r.raise_for_status()
        data = r.json()

        with col2:
            st.subheader("Результат")
            img_url = f"{API}{data['output_image']}"
            st.image(img_url, use_container_width=True)

        st.subheader("Найденные объекты")
        st.write(f"Request ID: {data['request_id']} | {data['processing_ms']} ms | detections: {data['num_detections']}")
        st.dataframe(data["detections"], use_container_width=True)

with tabs[1]:
    st.subheader("Сегментация (Mask R-CNN)")
    st.caption("Режимы визуализации: mask (наложение) или contour (контур). Маски в ответ API не отдаются, только has_mask.")
    col1, col2 = st.columns([1, 1])

    with col1:
        file2 = st.file_uploader("Выберите изображение для сегментации", type=["jpg", "jpeg", "png"], key="seg")
        conf2 = st.slider("Confidence (для детекции перед сегментацией)", 0.05, 0.95, 0.25, 0.05, key="seg_conf")
        only_backpack2 = st.checkbox("Показывать только backpack", value=True, key="seg_bp")
        mode = st.selectbox("Режим", ["mask", "contour"], index=0, key="seg_mode")
        run2 = st.button("Сегментировать", key="seg_run")

    if run2 and file2 is not None:
        files = {"file": (file2.name, file2.getvalue(), file2.type)}
        params = {"conf": conf2, "only_backpack": only_backpack2, "mode": mode}
        r = session.post(f"{API}/infer/segment", files=files, params=params, timeout=240)
        r.raise_for_status()
        data = r.json()

        with col2:
            st.subheader("Результат")
            img_url = f"{API}{data['output_image']}"
            st.image(img_url, use_container_width=True)

        st.subheader("Объекты")
        st.write(f"Request ID: {data['request_id']} | {data['processing_ms']} ms | detections: {data['num_detections']}")
        st.dataframe(data["detections"], use_container_width=True)

with tabs[2]:
    st.subheader("История запросов")
    limit = st.slider("Сколько показывать", 10, 200, 50, 10, key="hist_lim")
    r = session.get(f"{API}/history", params={"limit": limit}, timeout=60)
    r.raise_for_status()
    items = r.json()["items"]
    st.dataframe(items, use_container_width=True)

with tabs[3]:
    st.subheader("Отчёты")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Excel** (Requests + Detections)")
        st.markdown(f"[Скачать backpack_report.xlsx]({API}/report/excel)")

    with c2:
        st.markdown("**PDF** (сводка)")
        st.markdown(f"[Скачать backpack_report.pdf]({API}/report/pdf)")
