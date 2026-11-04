"""Cats vs. Dogs V3 — LOCAL EXPERIMENTAL PREVIEW.

Uses an ImageNet MobileNetV2 top-1 category gate before the existing VGG16
cat/dog classifier. NOT validated for public deployment. Keep app.py unchanged.
Requires models/cats_dogs_vgg16.keras and internet access for the first
MobileNetV2 model weights / ImageNet class index download.
"""

from io import BytesIO
from pathlib import Path
import warnings

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

MODEL_PATH = Path(__file__).resolve().parent / "models" / "cats_dogs_vgg16.keras"
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 32_000_000  # Cap decoded image size for cloud safety.
MAX_IMAGE_EDGE = 1600  # Downsize large photos before model inference.
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

st.set_page_config(
    page_title="Cats vs. Dogs V3 — Experimental Preview",
    page_icon="🐾",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading trained VGG16 model...")
def load_trained_model():
    import tensorflow as tf

    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    if tuple(model.input_shape[1:]) != (224, 224, 3):
        raise ValueError(
            f"Expected model input (224, 224, 3), but got {model.input_shape}."
        )
    return model



# Exact same ImageNet WordNet IDs and class-range rule as stage3_gate_probe.py.
# Top-1 is a conservative proxy, not a reliable open-set/OOD detector.
DOMESTIC_CATS = {
    "n02123045",  # tabby
    "n02123159",  # tiger_cat (a domestic cat coat pattern)
    "n02123394",  # Persian_cat
    "n02123597",  # Siamese_cat
    "n02124075",  # Egyptian_cat
}
DOG_MIN = 2085620  # n02085620 Chihuahua
DOG_MAX = 2113978  # n02113978 Mexican_hairless


def is_supported_imagenet_class(class_id: str) -> bool:
    if class_id in DOMESTIC_CATS:
        return True
    return (
        class_id.startswith("n")
        and len(class_id) == 9
        and class_id[1:].isdigit()
        and DOG_MIN <= int(class_id[1:]) <= DOG_MAX
    )


@st.cache_resource(show_spinner="Loading experimental MobileNetV2 gate...")
def load_gate_model():
    import tensorflow as tf
    return tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), weights="imagenet", include_top=True
    )


def inspect_input(model, image: Image.Image) -> tuple[bool, str, float]:
    """Use the exact same preprocessing and ImageNet top-1 decision as the probe."""
    import tensorflow as tf

    pixels = tf.convert_to_tensor(np.asarray(image, dtype=np.uint8))
    resized = tf.image.resize(pixels, (224, 224), method="bilinear")
    prepared = tf.keras.applications.mobilenet_v2.preprocess_input(resized)
    probabilities = np.asarray(model(tf.expand_dims(prepared, axis=0), training=False))
    class_id, label, score = tf.keras.applications.mobilenet_v2.decode_predictions(
        probabilities, top=1
    )[0][0]
    return is_supported_imagenet_class(class_id), str(label), float(score)


def decode_uploaded_image(data: bytes) -> Image.Image:
    """Validate uploads and safely downscale high-resolution images.

    The model still performs its original 224x224 VGG16 preprocessing.
    """
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("Image exceeds the 10 MB upload limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as test_image:
                if test_image.format not in ALLOWED_FORMATS:
                    raise ValueError("Please upload a JPEG, PNG, or WEBP image.")
                if test_image.width * test_image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds the 32-megapixel safety limit.")
                test_image.verify()

            with Image.open(BytesIO(data)) as image:
                # JPEG decoder can subsample before expanding large photos.
                image.draft("RGB", (MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
                corrected = ImageOps.exif_transpose(image)
                corrected.thumbnail(
                    (MAX_IMAGE_EDGE, MAX_IMAGE_EDGE),
                    Image.Resampling.LANCZOS,
                )
                rgb = corrected.convert("RGB")
                rgb.load()
                return rgb
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError(
            "This image cannot be decoded safely. Try another JPG, PNG, or WEBP."
        ) from exc


def dog_probability(model, image: Image.Image) -> float:
    """Exactly match the training notebook: tf.image.resize + VGG16 preprocess_input.

    Keras' VGG preprocessing expects 0..255 RGB, swaps to BGR, and subtracts
    ImageNet channel means. Do NOT divide pixels by 255 for this model.
    Output sigmoid probability of class 1 = 'dog' (class 0 = 'cat').
    """
    import tensorflow as tf

    pixels = tf.convert_to_tensor(np.asarray(image, dtype=np.uint8))
    resized = tf.image.resize(pixels, (224, 224), method="bilinear")
    preprocessed = tf.keras.applications.vgg16.preprocess_input(resized)
    batch = tf.expand_dims(preprocessed, axis=0)

    score = float(np.asarray(model(batch, training=False)).reshape(-1)[0])
    if not np.isfinite(score) or score < 0 or score > 1:
        raise ValueError("Model returned an invalid probability.")
    return score


st.title("🐾 Cats vs. Dogs V3 — Preview")
st.markdown("### Experimental two-model classification")
st.write(
    "Upload a photograph to see whether a VGG16-based classifier predicts "
    "**cat** or **dog**. The model was trained using transfer learning on "
    "the TensorFlow Datasets Cats vs. Dogs dataset."
)
st.warning(
    "**Experimental local preview — not validated for deployment.** "
    "A separate MobileNetV2 model first checks whether its top ImageNet label "
    "is a domestic cat or dog. It can reject real cats/dogs or accept unrelated images. "
    "Rejection means **uncertain / possibly unsupported**, not a verified identity."
)

with st.expander("How this model works", expanded=False):
    st.markdown(
        "**Architecture:** ImageNet-pretrained VGG16 (frozen backbone) + "
        "Flatten + Dropout(0.3) + Dense(1, sigmoid).  "
        "**Input:** RGB image resized to 224 × 224 using TensorFlow, then "
        "processed with `tf.keras.applications.vgg16.preprocess_input`.  "
        "**Classes:** cat = 0, dog = 1."
    )

model_available = MODEL_PATH.is_file()
if not model_available:
    st.warning(
        "The trained model file has not been added yet. Train and export "
        "`cats_dogs_vgg16.keras` using the included export notebook, "
        "then place it in `models/` to enable predictions."
    )

uploaded = st.file_uploader(
    "Choose a cat or dog image",
    type=["jpg", "jpeg", "png", "webp"],
    help="JPEG, PNG, or WEBP; up to 10 MB and 32 megapixels. Large photos are resized automatically.",
)

if uploaded is not None:
    try:
        image = decode_uploaded_image(uploaded.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)
    if not model_available:
        st.info("Add the exported trained model to get a real prediction.")
        st.stop()

    try:
        with st.spinner("Checking if this looks like a domestic cat or dog..."):
            gate_model = load_gate_model()
            gate_accepts, imagenet_label, imagenet_score = inspect_input(gate_model, image)
    except Exception as exc:
        st.error("The experimental detector could not run. No classification was made.")
        with st.expander("Technical details"):
            st.code(str(exc))
        st.stop()

    with st.expander("Experimental input-gate details", expanded=False):
        st.write(f"MobileNetV2 top ImageNet label: `{imagenet_label}`")
        st.write(f"Top-class model score: {imagenet_score:.1%}")
        st.caption("This score is NOT the probability the image belongs to a "
                   "supported class. The gate uses the top category only.")

    if not gate_accepts:
        st.warning(
            "⚠️ **Uncertain / possibly unsupported image.** "
            "The experimental gate did not identify a domestic cat or dog, "
            "so VGG16 classification was skipped. This could also be a "
            "false rejection of a real cat or dog."
        )
        st.stop()

    st.success("✅ Experimental gate accepted the input as a possible domestic cat/dog.")
    try:
        model = load_trained_model()
        with st.spinner("Classifying image with VGG16..."):
            dog_prob = dog_probability(model, image)
    except Exception as exc:
        st.error("Model inference failed. Check that the exported file is a valid VGG16 .keras model.")
        with st.expander("Technical details"):
            st.code(str(exc))
        st.stop()

    cat_prob = 1.0 - dog_prob
    predicted = "Dog 🐶" if dog_prob >= 0.5 else "Cat 🐱"
    selected_class_score = max(cat_prob, dog_prob)

    st.subheader("Binary classification result")
    st.write(f"**Model selected:** {predicted}")
    st.metric("Selected-class model score", f"{selected_class_score:.1%}")
    st.warning(
        "**Important:** This score does not guarantee correctness. "
        "MobileNetV2 accepted this image, but the gate can still make mistakes. "
        "The result is a forced two-class VGG16 prediction, not a calibrated "
        "probability of the animal truly appearing in the photograph."
    )
    st.write("**Scores across the two trained classes**")
    st.write(f"🐱 Cat — {cat_prob:.1%}")
    st.progress(cat_prob)
    st.write(f"🐶 Dog — {dog_prob:.1%}")
    st.progress(dog_prob)
    st.caption(
        "These are outputs from a two-class sigmoid classifier, not calibrated "
        "probabilities for arbitrary images. The experimental gate does not "
        "verify that a cat or dog is actually present."
    )

st.divider()
st.caption(
    "LOCAL EXPERIMENT ONLY • Transfer learning with TensorFlow/Keras • "
    "Images are processed for the current prediction; this app does not intentionally store uploads."
)
