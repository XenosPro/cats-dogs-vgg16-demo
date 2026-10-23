# Cats vs. Dogs — Live Transfer Learning Demo 🐾

A Streamlit image classifier powered by **your trained Keras VGG16 transfer-learning model**.

> **This starter does not contain trained weights.** The uploaded source notebook
> recorded a VGG16 transfer-learning experiment (97.56% test accuracy on its
> 450-image held-out subset), but it did not save a model file. To get real
> predictions, first export a trained `.keras` model (instructions below).

## 1. Export a real trained model (Kaggle recommended)

1. Open https://www.kaggle.com/code and choose **New Notebook**.
2. In the editor, choose **File → Import Notebook**, then upload **`export_vgg_kaggle.ipynb`**.
3. Set **Settings → Accelerator → GPU T4 x2** and switch **Internet → On** (needed for TFDS data and VGG16 ImageNet weights).
4. Select **Run All** and complete training, evaluation, and export.
5. Download **`cats_dogs_vgg16.keras`** and **`test_metrics.json`** from `/kaggle/working/` using the final notebook cell or Kaggle's output/files panel. You can also use **Save Version** to persist outputs.

Alternatively, open **`export_vgg_colab.ipynb`** in Colab and run its cells **in order**.
It reproduces the VGG16 transfer-learning experiment from the original notebook:

- Dataset: TensorFlow Datasets `cats_vs_dogs`, 1,500 images per class
- Split: 2,100 train / 450 validation / 450 test
- Model: `VGG16(weights="imagenet", include_top=False)` (frozen) → `Flatten` → `Dropout(0.3)` → `Dense(1, sigmoid)`
- Training: Adam 1e-3, binary cross-entropy, 8 epochs, batch size 32
- Inference preprocessing: resize with `tf.image.resize` to 224 × 224, then **VGG16 preprocess_input**. Do **not** scale inputs to 0–1.

The notebook evaluates the model, saves **`cats_dogs_vgg16.keras`**, reloads it,
and checks that its predictions match. Download the model file from Kaggle (or Colab) and move it to:

```text
cats-dogs-live-demo/
  models/
    cats_dogs_vgg16.keras
```

If an original Colab session is still open, you can instead run these lines
**immediately after successfully training and evaluating `vgg_tl_model`**:

```python
vgg_tl_model.save("cats_dogs_vgg16.keras")
from google.colab import files
files.download("cats_dogs_vgg16.keras")
```

**Important:** In the uploaded notebook, the VGG16 model-building cell has a
later execution count than its training cell. Running the build cell again
creates a **fresh untrained** model. Re-run the build → compile → train →
evaluate sequence in order before saving, or use the clean export notebook.

## 2. Run the web app locally (Windows PowerShell)

Install **Python 3.11** if you do not already have it.

```powershell
cd cats-dogs-live-demo
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use the environment's Python executable:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If the model file is missing, the app still opens but clearly states that
predictions are unavailable. It never makes up a prediction.

## 3. Deploy publicly on Streamlit Community Cloud

1. Push this folder (including the **trained model file**) to a new GitHub repo.
2. Open https://share.streamlit.io/ and choose to deploy from an existing repo.
3. Select `app.py` as the entrypoint and choose Python 3.11 in deployment settings.
4. Make sure `requirements.txt` is at the repository root.
5. After deployment, use the `*.streamlit.app` URL as the **Live Demo** link on your portfolio.

The full VGG16 file may be roughly 60 MB; GitHub warns about regular files over
50 MB and refuses files over 100 MB. If needed, use model hosting instead of
committing the weights directly. Community Cloud memory/build constraints also
apply; if the app cannot start, review the deployment logs and consider a
smaller trained MobileNetV2 model as a future upgrade.

## Notes on honest metrics and limitations

- The **97.56%** result was recorded in the uploaded notebook for **VGG16
  transfer learning**, not for any of the unexecuted fine-tuning architectures.
  Your **retrained** model's accuracy may differ.
- The app's confidence score is **not calibrated certainty**.
- This binary model always chooses cat or dog; it does not recognize out-of-domain images.
- The app processes uploaded images for prediction and does not implement
  intentional disk storage of uploads. Your host may maintain infrastructure logs.
- Do not upload private photographs or sensitive data to a public demo.

## File layout

```text
cats-dogs-live-demo/
├── app.py
├── export_vgg_kaggle.ipynb   # preferred: train/export in Kaggle
├── export_vgg_colab.ipynb    # alternative: train/export in Colab
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
└── models/
    ├── README.md
    └── cats_dogs_vgg16.keras  # add after exporting
```
