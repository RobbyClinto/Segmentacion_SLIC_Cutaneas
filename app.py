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
    orden = np.argsort(probs)[::-1]
    for idx in orden:
        class_name = CLASES[idx]
        prob = probs[idx]
        es_top = idx == orden[0]
        etiqueta = f"**{CLASES_UI[class_name]}**" if es_top else CLASES_UI[class_name]
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; font-size:0.85rem; "
            f"margin-top:6px;'><span>{etiqueta}</span><span>{prob:.1%}</span></div>",
            unsafe_allow_html=True,
        )
        st.progress(float(prob))


# ----------------------------------------------------------------------------------
# INTERFAZ
# ----------------------------------------------------------------------------------

def inyectar_estilos():
    st.markdown(
        """
        <style>
        .main-header {
            padding: 1.6rem 2rem;
            border-radius: 14px;
            background: linear-gradient(120deg, #0f2b46 0%, #1c4e73 55%, #2f7c8c 100%);
            color: #ffffff;
            margin-bottom: 1.2rem;
        }
        .main-header h1 {
            margin-bottom: 0.2rem;
            font-size: 1.9rem;
        }
        .main-header p {
            margin: 0;
            opacity: 0.9;
            font-size: 0.95rem;
        }
        .card {
            background: var(--background-color, #ffffff);
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 12px;
            padding: 1rem 1.1rem 1.2rem 1.1rem;
            height: 100%;
        }
        .card h4 {
            margin-top: 0;
            margin-bottom: 0.6rem;
            font-size: 1.05rem;
        }
        .badge {
            display: inline-block;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }
        .badge-blue { background: #e7f1fb; color: #14507d; }
        .legend-dot {
            width: 12px; height: 12px; display: inline-block;
            border-radius: 50%; margin-right: 6px; vertical-align: middle;
        }
        .footer-note {
            text-align: center;
            font-size: 0.8rem;
            opacity: 0.65;
            margin-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def mostrar_encabezado():
    st.markdown(
        """
        <div class="main-header">
            <h1>🔬 Diagnóstico asistido de lesiones cutáneas</h1>
            <p>Comparación entre un modelo CNN clásico y una variante con segmentación SLIC previa</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mostrar_sidebar():
    with st.sidebar:
        st.header("ℹ️ Acerca de esta demo")
        st.markdown(
            "Esta aplicación forma parte de un proyecto de investigación académica "
            "sobre segmentación de imágenes dermatoscópicas con **SLIC** "
            "(*Simple Linear Iterative Clustering*) combinada con redes neuronales "
            "convolucionales."
        )
        st.markdown("**Pipeline:**")
        st.markdown(
            "1. Segmentación en superpíxeles (SLIC)\n"
            "2. Clasificación de cada superpíxel (piel / lesión / artefacto)\n"
            "3. Limpieza de artefactos mediante inpainting\n"
            "4. Clasificación final con ambos modelos"
        )
        st.divider()
        st.subheader("Clases posibles")
        for clave, nombre in CLASES_UI.items():
            st.markdown(f"- **{nombre}**")
        st.divider()
        st.caption(
            "⚠️ Herramienta de investigación / demostración académica. "
            "No sustituye una evaluación médica profesional."
        )


def mostrar_leyenda_colores():
    st.markdown(
        "<div style='display:flex; gap:18px; align-items:center; margin-top:10px; flex-wrap: wrap;'>"
        "<span><span class='legend-dot' style='background:#00d26a;'></span>Piel</span>"
        "<span><span class='legend-dot' style='background:#ff4b4b;'></span>Lesión</span>"
        "<span><span class='legend-dot' style='background:#ffd23f;'></span>Artefacto</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def tarjeta_resultado(titulo: str, badge: str, imagen: np.ndarray, clase_predicha: str, probabilidad: float, probs: np.ndarray):
    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<span class='badge badge-blue'>{badge}</span>", unsafe_allow_html=True)
    st.markdown(f"#### {titulo}")
    st.image(Image.fromarray(imagen), width="stretch")
    st.metric(label="Clase predicha", value=clase_predicha, delta=f"{probabilidad:.1%} de confianza")
    with st.expander("Ver todas las probabilidades"):
        mostrar_barras_probabilidades(probs)
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Diagnóstico de lesiones cutáneas",
        page_icon="🔬",
        layout="wide",
    )
    inyectar_estilos()
    mostrar_sidebar()
    mostrar_encabezado()

    st.warning(
        "Esto es una herramienta de investigación / demostración académica, "
        "**NO** un diagnóstico médico."
    )

    uploaded_file = st.file_uploader(
        "📤 Sube una imagen dermatoscópica (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        help="La imagen se procesa localmente en esta sesión y no se almacena.",
    )

    if uploaded_file is None:
        st.info("👆 Sube una imagen para ejecutar el pipeline completo.")
        st.markdown(
            "<div class='footer-note'>Demo académica · Segmentación SLIC + CNN</div>",
            unsafe_allow_html=True,
        )
        return

    img_rgb = cargar_imagen(uploaded_file)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    progreso = st.progress(0, text="Iniciando pipeline...")

    with st.spinner("Ejecutando segmentación SLIC y clasificando superpíxeles..."):
        labels, segment_classes, _ = segmentar_y_clasificar_superpixeles(img_rgb)
        mapa_colores = pintar_mapa_colores(img_rgb, labels, segment_classes)
        imagen_limpia = limpiar_imagen(img_bgr, labels, segment_classes)
        imagen_limpia_rgb = cv2.cvtColor(imagen_limpia, cv2.COLOR_BGR2RGB)
    progreso.progress(40, text="Segmentación completa. Ejecutando modelo baseline...")

    with st.spinner("Prediciendo con el modelo baseline..."):
        probs_baseline, pred_baseline = predecir_baseline(img_rgb)
    progreso.progress(70, text="Baseline completo. Ejecutando modelo con SLIC...")

    with st.spinner("Prediciendo con el modelo propuesto con SLIC..."):
        probs_slic, pred_slic = predecir_slic(imagen_limpia_rgb)
    progreso.progress(100, text="¡Listo!")
    progreso.empty()

    st.markdown("### 🧩 Resultado de la segmentación")
    col_img, col_leyenda = st.columns([3, 1])
    with col_img:
        st.image(Image.fromarray(mapa_colores), width="stretch")
    with col_leyenda:
        st.markdown("**Leyenda**")
        mostrar_leyenda_colores()

    st.markdown("---")
    st.markdown("### 🧪 Comparación de modelos")

    col1, col2 = st.columns(2)
    with col1:
        tarjeta_resultado(
            titulo="CNN sin SLIC (baseline)",
            badge="Modelo base",
            imagen=img_rgb,
            clase_predicha=CLASES_UI[CLASES[pred_baseline]],
            probabilidad=float(probs_baseline[pred_baseline]),
            probs=probs_baseline,
        )
    with col2:
        tarjeta_resultado(
            titulo="CNN con SLIC (propuesto)",
            badge="Modelo propuesto",
            imagen=imagen_limpia_rgb,
            clase_predicha=CLASES_UI[CLASES[pred_slic]],
            probabilidad=float(probs_slic[pred_slic]),
            probs=probs_slic,
        )

    st.markdown("---")
    if CLASES[pred_baseline] == CLASES[pred_slic]:
        st.success(f"✅ Ambos modelos coinciden: **{CLASES_UI[CLASES[pred_baseline]]}**")
    else:
        st.warning(
            f"⚠️ Los modelos discrepan: baseline dice **{CLASES_UI[CLASES[pred_baseline]]}**, "
            f"SLIC dice **{CLASES_UI[CLASES[pred_slic]]}**. "
            "Esto puede deberse a distorsión por inpainting sobre la lesión, "
            "como en la Figura 4 del artículo."
        )

    st.markdown(
        "<div class='footer-note'>Demo académica · Segmentación SLIC + CNN</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
