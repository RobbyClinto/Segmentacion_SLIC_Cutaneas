import os
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image
from skimage.segmentation import slic
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


CLASES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASES_UI = {
    "akiec": "Queratosis actínica",
    "bcc": "Carcinoma basocelular",
    "bkl": "Lesión queratósica benigna",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Nevus melanocítico",
    "vasc": "Lesión vascular",
}

COLOR_MAP = {
    0: (0, 255, 0),      # piel - verde
    1: (255, 0, 0),      # lesión - rojo
    2: (255, 255, 0),    # artefacto - amarillo
}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource(show_spinner=False)
def cargar_modelo(ruta: str):
    return tf.keras.models.load_model(ruta)


def resolver_ruta_modelo(nombre_archivo: str) -> str:
    rutas = [
        os.path.join(ROOT_DIR, "models", nombre_archivo),
        os.path.join(ROOT_DIR, nombre_archivo),
    ]
    for ruta in rutas:
        if os.path.exists(ruta):
            return ruta
    raise FileNotFoundError(f"No se encontró el modelo {nombre_archivo} en {rutas}")


def cargar_imagen(uploaded_file) -> np.ndarray:
    img_pil = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(img_pil)
    return img_rgb


def segmentar_y_clasificar_superpixeles(img_rgb: np.ndarray, umbral_artefacto: float = 0.8):
    """
    Segmenta la imagen con SLIC y clasifica cada superpíxel en piel/lesión/artefacto.

    IMPORTANTE: el parche de cada superpíxel se enmascara (se pone en negro todo
    lo que no pertenece al superpíxel) ANTES de redimensionar, exactamente igual
    que en el entrenamiento (ver extraer_parche_superpixel en el notebook).
    Sin este enmascarado, el modelo recibe parches "contaminados" con contenido
    de superpíxeles vecinos, lo cual no coincide con la distribución de datos
    vista durante el entrenamiento y degrada las predicciones.
    """
    img_rgb = img_rgb.astype(np.uint8)
    labels = slic(img_rgb, n_segments=100, compactness=10, sigma=1, start_label=1)

    modelo = cargar_modelo(resolver_ruta_modelo("cnn_filtradora_final.h5"))

    segment_classes = {}
    segment_probs = {}

    for segment_id in np.unique(labels):
        mask = labels == segment_id
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue

        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1

        patch = img_rgb[y0:y1, x0:x1].copy()
        mask_patch = mask[y0:y1, x0:x1]
        patch[~mask_patch] = 0  # aislar el superpíxel, igual que en entrenamiento

        if patch.size == 0:
            continue

        patch = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_LINEAR)
        patch = patch.astype(np.float32) / 255.0
        patch = np.expand_dims(patch, axis=0)

        probs = modelo.predict(patch, verbose=0)[0]
        pred = int(np.argmax(probs))

        if pred == 2 and probs[2] < umbral_artefacto:
            pred = int(np.argmax([probs[0], probs[1]]))

        segment_classes[int(segment_id)] = pred
        segment_probs[int(segment_id)] = probs

    return labels, segment_classes, segment_probs


def pintar_mapa_colores(img_rgb: np.ndarray, labels: np.ndarray, segment_classes: dict):
    img_rgb = img_rgb.astype(np.float32) / 255.0
    overlay = img_rgb.copy()
    alpha = 0.45

    for segment_id in np.unique(labels):
        cls_idx = segment_classes.get(int(segment_id), 0)
        color = np.array(COLOR_MAP[cls_idx], dtype=np.float32) / 255.0
        mask = labels == segment_id
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

    return (overlay * 255.0).astype(np.uint8)


def limpiar_imagen(img_rgb: np.ndarray, labels: np.ndarray, segment_classes: dict):
    img_rgb = img_rgb.astype(np.uint8)
    mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    for segment_id in np.unique(labels):
        if segment_classes.get(int(segment_id), 0) == 2:
            mask[labels == segment_id] = 255

    if np.count_nonzero(mask) == 0:
        return img_rgb

    mask_blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 15.0)
    mask_blurred = np.clip(mask_blurred, 0, 255).astype(np.uint8)

    cleaned = cv2.inpaint(img_rgb, mask_blurred, 3, cv2.INPAINT_NS)
    return cleaned


def predecir_baseline(img_rgb: np.ndarray):
    modelo = cargar_modelo(resolver_ruta_modelo("mobilenet_baseline.h5"))

    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = preprocess_input(img_resized.astype(np.float32))
    x = np.expand_dims(x, axis=0)
    probs = modelo.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return probs, pred_idx


def predecir_slic(img_rgb: np.ndarray):
    modelo = cargar_modelo(resolver_ruta_modelo("modelo_final_CON_slic.h5"))

    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = preprocess_input(img_resized.astype(np.float32))
    x = np.expand_dims(x, axis=0)
    probs = modelo.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return probs, pred_idx


def mostrar_barras_probabilidades(probs: np.ndarray):
    for class_name, prob in zip(CLASES, probs):
        st.markdown(f"- **{CLASES_UI[class_name]}**: {prob:.2%}")
        st.progress(float(prob))


def main():
    st.set_page_config(page_title="Demo diagnóstico de lesiones cutáneas", layout="wide")
    st.title("Demo de diagnóstico de lesiones cutáneas con SLIC")
    st.warning("Esto es una herramienta de investigación / demostración académica, NO un diagnóstico médico.")

    uploaded_file = st.file_uploader("Sube una imagen dermatoscópica (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Sube una imagen para ejecutar el pipeline completo.")
        return

    img_rgb = cargar_imagen(uploaded_file)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    with st.spinner("Ejecutando segmentación SLIC y clasificando superpíxeles..."):
        labels, segment_classes, _ = segmentar_y_clasificar_superpixeles(img_rgb)
        mapa_colores = pintar_mapa_colores(img_rgb, labels, segment_classes)
        imagen_limpia = limpiar_imagen(img_bgr, labels, segment_classes)
        imagen_limpia_rgb = cv2.cvtColor(imagen_limpia, cv2.COLOR_BGR2RGB)

    with st.spinner("Prediciendo con el modelo baseline..."):
        probs_baseline, pred_baseline = predecir_baseline(img_rgb)

    with st.spinner("Prediciendo con el modelo propuesto con SLIC..."):
        probs_slic, pred_slic = predecir_slic(imagen_limpia_rgb)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Segmentación SLIC")
        st.image(Image.fromarray(mapa_colores), use_container_width=True)
        st.markdown("<div style='display:flex; gap:12px; align-items:center; margin-top:8px;'>"
                    "<span style='width:14px; height:14px; background:#00ff00; display:inline-block; border-radius:50%;'></span> Piel  "
                    "<span style='width:14px; height:14px; background:#ff0000; display:inline-block; border-radius:50%;'></span> Lesión  "
                    "<span style='width:14px; height:14px; background:#ffff00; display:inline-block; border-radius:50%;'></span> Artefacto"
                    "</div>", unsafe_allow_html=True)

    with col2:
        st.subheader("CNN SIN SLIC (baseline)")
        st.image(Image.fromarray(img_rgb), use_container_width=True)
        st.markdown(f"**Clase predicha:** {CLASES_UI[CLASES[pred_baseline]]}")
        st.markdown(f"**Probabilidad:** {probs_baseline[pred_baseline]:.2%}")
        mostrar_barras_probabilidades(probs_baseline)

    with col3:
        st.subheader("CNN CON SLIC (propuesto)")
        st.image(Image.fromarray(imagen_limpia_rgb), use_container_width=True)
        st.markdown(f"**Clase predicha:** {CLASES_UI[CLASES[pred_slic]]}")
        st.markdown(f"**Probabilidad:** {probs_slic[pred_slic]:.2%}")
        mostrar_barras_probabilidades(probs_slic)

    st.markdown("---")
    if CLASES[pred_baseline] == CLASES[pred_slic]:
        st.success(f"✅ Ambos modelos coinciden: {CLASES_UI[CLASES[pred_baseline]]}")
    else:
        st.warning(
            f"⚠️ Los modelos discrepan: baseline dice {CLASES_UI[CLASES[pred_baseline]]}, SLIC dice {CLASES_UI[CLASES[pred_slic]]}."
            " Esto puede deberse a distorsión por inpainting sobre la lesión, como en la Figura 4 del artículo."
        )


if __name__ == "__main__":
    main()
