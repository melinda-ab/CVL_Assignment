import json

notebook = {
    "cells": [],
    "metadata": {
      "kernelspec": {
       "display_name": "Python 3",
       "language": "python",
       "name": "python3"
      },
      "language_info": {
       "codemirror_mode": {
        "name": "ipython",
        "version": 3
       },
       "file_extension": ".py",
       "mimetype": "text/x-python",
       "name": "python",
       "nbconvert_exporter": "python",
       "pygments_lexer": "ipython3",
       "version": "3.8.0"
      }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_markdown(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")]
    })

add_markdown("# Full Pipeline: Road Sign Detection\nSesuai dengan instruksi, notebook ini menggabungkan dataset loader, evaluasi, dan modul utama menjadi satu.\nDataset juga akan diunduh dari Kaggle sehingga dapat dijalankan oleh siapa saja tanpa memerlukan dataset lokal.")

add_markdown("## 1. Unduh dan Ekstrak Dataset (Kaggle)\nStruktur akan mengikuti format:\n- gtsrb/Meta\n- gtsrb/Test\n- gtsrb/Train\n- gtsrb/Meta.csv\n- gtsrb/Test.csv\n- gtsrb/Train.csv")
add_code("!pip install kaggle\n\n# Perintah untuk mengunduh dataset. Pastikan Anda memiliki kaggle.json yang valid dikonfigurasi di /.kaggle/kaggle.json\n!kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign\n\n# Ekstrak file zip ke folder gtsrb\n!unzip -q -o gtsrb-german-traffic-sign.zip -d gtsrb")

with open('dataset_loader.py', 'r', encoding='utf-8') as f:
    dl_code = f.read()
add_markdown("## 2. Dataset Loader (`dataset_loader.py`)")
add_code(dl_code)

with open('evaluation.py', 'r', encoding='utf-8') as f:
    ev_code = f.read()
add_markdown("## 3. Evaluation (`evaluation.py`)")
add_code(ev_code)

with open('detector.py', 'r', encoding='utf-8') as f:
    dt_code = f.read()
add_markdown("## 4. Detector Pipeline (`detector.py`)")

# Sedikit penyesuaian: Menghilangkan import dataset_loader dan evaluation karena sudah ada di sel atasnya.
# Atau sesuai instruksi "jangan ada yang diubah selain instruksi saya", kita akan menyertakan kode aslinya 
# namun mengomentari bagian importnya agar bisa running di dalam satu notebook.
dt_code_mod = dt_code.replace("import dataset_loader as dl", "# import dataset_loader as dl")
dt_code_mod = dt_code_mod.replace("import evaluation as ev", "# import evaluation as ev")

# Mengganti referensi 'dl.' dan 'ev.' dengan nama aslinya karena sudah ada di scope global Notebook.
dt_code_mod = dt_code_mod.replace("dl.", "")
dt_code_mod = dt_code_mod.replace("ev.", "")

add_code(dt_code_mod)

with open('full–pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook berhasil dibuat: full–pipeline.ipynb")
