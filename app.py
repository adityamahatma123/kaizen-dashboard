import difflib
import io
import json
import os
import re
import tempfile
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

import pandas as pd
import streamlit as st

# ==========================================
# 1. KONFIGURASI HALAMAN & TEMA (UI/UX)
# ==========================================
# st.set_page_config WAJIB jadi perintah Streamlit pertama.
st.set_page_config(page_title="Portal Validasi Kaizen", page_icon="🏢", layout="wide")

# Import library AI dibuat setelah set_page_config + dibungkus try,
# supaya kalau paket belum terpasang muncul pesan jelas (bukan loading tanpa akhir).
try:
    from google import genai
    from google.genai import types
    from groq import Groq
except ImportError as e:
    st.error(
        f"Library belum terpasang: `{e}`. Pastikan file `requirements.txt` di GitHub berisi "
        "`google-genai`, `groq`, `pandas`, `xlsxwriter`, lalu Reboot app."
    )
    st.stop()

st.markdown(
    """
    <style>
    /* Header JANGAN disembunyikan: indikator "Running..." ada di sana.
       Kalau disembunyikan, app yang sedang bekerja terlihat seperti hang. */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #5C7C99; text-align: center; font-family: 'Nunito', 'Segoe UI', sans-serif; font-weight: 700; margin-bottom: 0.5rem;}

    .stButton>button {
        background-color: #A3B9D2; color: white; border-radius: 8px; border: none;
        padding: 0.6rem 1.2rem; font-weight: 600; transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stButton>button:hover {
        background-color: #8BA3C7; color: white; transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🏢 Portal Validasi Kaizen - Dept. MEX")
st.markdown(
    "<p style='text-align: center; color: #7F8C8D; font-size: 1.1rem; font-weight: 400; margin-bottom: 2rem;'>Unggah"
    " Sistem evaluasi dokumen, Gemini vs Groq. Dokumen yang diupload tidak akan disimpan di database, jadi pastikan output data sudah di download secara manual sebelum menutup aplikasi ini.</p>",
    unsafe_allow_html=True,
)    
st.divider()

# Kompatibilitas parameter lebar tabel antar versi Streamlit
def _versi_mayor_minor(v):
    angka = re.findall(r"\d+", v)[:2]
    return tuple(int(x) for x in angka) if len(angka) == 2 else (0, 0)

LEBAR = {"width": "stretch"} if _versi_mayor_minor(st.__version__) >= (1, 50) else {"use_container_width": True}

# ==========================================
# 2. KONFIGURASI, API KEY & KLIEN
# ==========================================
def _secret(nama, default=None):
    try:
        return str(st.secrets[nama]).strip()
    except Exception:
        return os.environ.get(nama, default)

API_KEY_GEMINI = _secret("GEMINI_API_KEY")
API_KEY_GROQ = _secret("GROQ_API_KEY")
_hilang = [n for n, v in (("GEMINI_API_KEY", API_KEY_GEMINI), ("GROQ_API_KEY", API_KEY_GROQ)) if not v]
if _hilang:
    st.error(f"API Key belum diisi di Secrets: {', '.join(_hilang)}. Isi di Settings → Secrets, lalu Reboot app.")
    st.stop()

# Model bisa diganti lewat Secrets tanpa edit kode (GEMINI_MODEL, GROQ_MODEL)
MODEL_GEMINI = _secret("GEMINI_MODEL", "gemini-3.5-flash-lite")
MODEL_GROQ = _secret("GROQ_MODEL", "openai/gpt-oss-120b")
THINKING_LEVEL = _secret("THINKING_LEVEL", "medium")
COOLDOWN_GEMINI = int(_secret("COOLDOWN_GEMINI", "5"))  # detik; naikkan jika sering kena 429
BATAS_UPLOAD_DETIK = 240
# --- Pengaturan Groq (free tier gpt-oss-120b: 8K token/menit, 200K token/hari) ---
GROQ_TPM = int(_secret("GROQ_TPM", "7500"))                 # target token/menit (sedikit di bawah limit 8000)
GROQ_TPD = int(_secret("GROQ_TPD", "200000"))               # limit token/hari (untuk penghitung di sidebar)
GROQ_MAX_OUTPUT = int(_secret("GROQ_MAX_OUTPUT", "3000"))   # cadangan token output per panggilan
GROQ_CHAR_PER_TOKEN = float(_secret("GROQ_CHAR_PER_TOKEN", "3.2"))  # estimasi karakter per token
GROQ_REASONING = _secret("GROQ_REASONING", "low")           # low/medium/high (khusus gpt-oss)
GROQ_EXTRA = {"extra_body": {"reasoning_effort": GROQ_REASONING}} if "gpt-oss" in MODEL_GROQ else {}


@st.cache_resource(show_spinner=False)
def _buat_klien_gemini(api_key):
    return genai.Client(api_key=api_key)


@st.cache_resource(show_spinner=False)
def _buat_klien_groq(api_key):
    return Groq(api_key=api_key, timeout=120, max_retries=0)


client_gemini = _buat_klien_gemini(API_KEY_GEMINI)
client_groq = _buat_klien_groq(API_KEY_GROQ)


class GroqPacer:
    """Pengatur laju token Groq: menunggu secukupnya agar total token dalam 60 detik terakhir tidak melewati limit."""

    def __init__(self, tpm):
        self.tpm = tpm
        self.kunci = threading.Lock()
        self.riwayat = deque()  # entri: [waktu, jumlah_token]
        self.harian = {}
        self._habis_sampai = 0.0

    @staticmethod
    def _hari():
        return datetime.now().strftime("%Y-%m-%d")

    def tunggu(self, estimasi, log):
        sudah_log = False
        while True:
            with self.kunci:
                sekarang = time.time()
                while self.riwayat and sekarang - self.riwayat[0][0] >= 60:
                    self.riwayat.popleft()
                terpakai = sum(e[1] for e in self.riwayat)
                if not self.riwayat or terpakai + estimasi <= self.tpm:
                    entri = [sekarang, estimasi]
                    self.riwayat.append(entri)
                    return entri
                tunggu = 60 - (sekarang - self.riwayat[0][0]) + 0.5
            if not sudah_log:
                log.write(f"⏱️ Groq: menunggu ±{tunggu:.0f} detik agar tidak melewati limit token/menit...")
                sudah_log = True
            time.sleep(max(1.0, min(tunggu, 15.0)))

    def catat(self, entri, aktual):
        with self.kunci:
            entri[0] = time.time()
            entri[1] = aktual
            self.harian[self._hari()] = self.harian.get(self._hari(), 0) + aktual

    def batal(self, entri):
        with self.kunci:
            entri[1] = 0

    def tandai_habis(self):
        self._habis_sampai = time.time() + 1800  # coba lagi setelah 30 menit

    def habis(self):
        return time.time() < self._habis_sampai

    def pemakaian_hari_ini(self):
        return self.harian.get(self._hari(), 0)


@st.cache_resource(show_spinner=False)
def _buat_pacer(tpm):
    return GroqPacer(tpm)


PACER = _buat_pacer(GROQ_TPM)


def buat_config(json_mode=False):
    """Config Gemini yang aman untuk versi SDK lama (kalau thinking_level tidak didukung)."""
    kwargs = {"seed": 42}
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    try:
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL), **kwargs
        )
    except Exception:
        return types.GenerateContentConfig(**kwargs)


CONFIG_TEXT = buat_config(False)
CONFIG_JSON = buat_config(True)


def _v(paket):
    try:
        return version(paket)
    except PackageNotFoundError:
        return "tidak terpasang"


with st.sidebar.expander("🔧 Diagnostik"):
    st.write(f"streamlit: `{st.__version__}`")
    st.write(f"google-genai: `{_v('google-genai')}`")
    st.write(f"groq: `{_v('groq')}`")
    st.write(f"Model Gemini: `{MODEL_GEMINI}`")
    st.write(f"Model Groq: `{MODEL_GROQ}`")
    _pakai = PACER.pemakaian_hari_ini()
    st.write(f"Token Groq terpakai hari ini (perkiraan): `{_pakai:,}` / `{GROQ_TPD:,}`")
    st.progress(min(_pakai / max(GROQ_TPD, 1), 1.0))

# ==========================================
# 3. DATA RUBRIK
# ==========================================
KRITERIA_RUJUKAN_VALIDASI_MANUAL = {
    7: "Pemetaan 4M — verifikasi kesesuaian dengan kondisi mesin/area aktual di lapangan",
    10: "Ketepatan Root Cause — memerlukan justifikasi teknis dari asesor lapangan",
    11: "Action Plan & PIC — konfirmasi PIC dan jadwal riil pelaksanaan",
    18: "Kelengkapan Standardisasi — pengecekan dokumen fisik/SOP terbaru di lokasi kerja",
    19: "Validasi Standardisasi — verifikasi implementasi standar di lapangan",
    21: "Replikasi — konfirmasi area lain yang benar-benar direplikasi",
}
_DAFTAR_RUJUKAN_VALIDASI_STR = "\n".join(
    f"- Kriteria {no}: {alasan}" for no, alasan in KRITERIA_RUJUKAN_VALIDASI_MANUAL.items()
)

# no: (tahap, nama kriteria, skor yang diperbolehkan)
RUBRIK_META = {
    1: ("PLAN", "5G", {0, 1, 2}),
    2: ("PLAN", "Losses Measurement", {0, 1, 2}),
    3: ("PLAN", "Kelengkapan 5W1H", {0, 1, 2}),
    4: ("PLAN", "Visualisasi/Sketch Fenomena", {0, 1, 2}),
    5: ("PLAN", "Target SMART", {0, 2}),
    6: ("PLAN", "Fishbone Diagram/4M", {0, 1, 2}),
    7: ("PLAN", "Pemetaan 4M pada Fishbone", {0, 1, 2}),
    8: ("PLAN", "Hubungan Akar Penyebab (Why-Why)", {0, 1, 2}),
    9: ("PLAN", "Bukti Akar Penyebab", {0, 3, 5}),
    10: ("PLAN", "Ketepatan Root Cause", {0, 1, 2}),
    11: ("PLAN", "Action Plan & PIC", {0, 1, 2}),
    12: ("PLAN", "Rencana Perbaikan per Sumber Masalah", {0, 1, 2}),
    13: ("PLAN", "Form Usulan Perbaikan (FUP)", {0, 3, 5}),
    14: ("DO", "Pelaksanaan Action Plan", {0, 1, 2}),
    15: ("DO", "Dokumentasi Pelaksanaan", {0, 5, 8}),
    16: ("CHECK", "Pencapaian Target", {0, 1}),
    17: ("CHECK", "Pengecekan Hasil", {0, 3, 5}),
    18: ("ACT", "Kelengkapan Standardisasi", {0, 3, 5}),
    19: ("ACT", "Validasi Standardisasi", {0, 1, 2}),
    20: ("ACT", "Tindak Lanjut Sosialisasi", {0, 3, 5}),
    21: ("ACT", "Replikasi ke Area/Mesin Lain", {0, 3, 5}),
}

RUBRIK_21_POIN_DETAIL = """
TAHAP PLAN — 1. Definisikan Masalah & Tentukan Target
1. 5G — 0: Tidak ada evidence | 1: Ada evidence tapi tidak relevan dengan masalah | 2: Ada evidence dan relevan dengan masalah
2. Losses Measurement (bagian dari 5W1H) — 0: Tidak ada losses measurement masalah | 1: Ada tapi tidak relevan dengan masalah | 2: Ada dan relevan dengan masalah
3. Kelengkapan 5W1H — 0: Tidak memenuhi semua kriteria 5W1H | 1: Sebagian memenuhi kriteria 5W1H | 2: Memenuhi semua kriteria 5W1H
4. Visualisasi/Sketch fenomena — 0: Tidak ada penjabaran dan fenomena | 1: Ada tapi tidak relevan dengan masalah | 2: Ada dan relevan dengan masalah
5. Target SMART — 0: Tidak memenuhi SMART & tidak sejalan dengan deskripsi masalah | 2: Semua penjabaran memenuhi SMART & sejalan dengan deskripsi masalah (kriteria ini TIDAK punya opsi skor 1)

TAHAP PLAN — 2. Klasifikasi Potensi Sumber Masalah
6. Fishbone Diagram/4M — 0: Tidak memiliki fishbone diagram/4M | 1: Fishbone dibuat namun belum lengkap/analisa dangkal | 2: Fishbone dibuat dengan sistematis dan lengkap
7. Pemetaan 4M pada fishbone — 0: Pemetaan 4M belum tepat pada fishbone | 1: Sebagian pemetaan 4M sudah tepat | 2: Pemetaan 4M sudah SELURUHNYA tepat
   WAJIB PERIKSA MENDETAIL: untuk SETIAP cabang/duri pada fishbone, tentukan apakah penyebab itu ditempatkan pada kategori 4M yang BENAR (Man/Method/Machine/Material). Contoh kesalahan umum: penyebab soal alat/mesin dimasukkan ke kategori Man, atau penyebab soal prosedur dimasukkan ke Material. Sebutkan eksplisit kalau menemukan kesalahan kategori.

TAHAP PLAN — 3. Deteksi Sumber Masalah
8. Hubungan Akar Penyebab (Why-Why Analysis / 5 Whys) — 0: Hubungan akar penyebab tidak relevan dan tidak terkait | 1: Sebagian hubungan akar penyebab saling terkait dan benar | 2: Semua hubungan akar penyebab saling terkait dan benar
   WAJIB PERIKSA RANTAI LOGIKA 5 WHYS SECARA TEKSTUAL, bukan cuma cek "ada 5 baris why": baca isi tiap why satu per satu, verifikasi apakah jawaban why ke-N benar-benar PENYEBAB LANGSUNG dari why ke-(N-1). Kalau ada loncatan logika / why yang tidak nyambung, turunkan skor meski jumlah why-nya lengkap 5. Verifikasi juga apakah akar masalah akhir dari rantai why ini benar-benar terhubung ke salah satu cabang/duri di fishbone (kepala ikan/masalah utama) — kalau topik akhirnya tidak muncul sama sekali di fishbone, itu indikasi inkonsistensi antar tools dan HARUS disebutkan. Pertimbangkan juga apakah ada kemungkinan faktor 4M lain yang relevan tapi terlewat/tidak dieksplorasi sama sekali.
9. Bukti Akar Penyebab — 0: Tidak ada bukti (data pendukung/visualisasi) | 3: Sebagian akar penyebab dapat dibuktikan melalui dokumen pendukung/report trial/visual | 5: Semua akar penyebab dapat dibuktikan
10. Ketepatan Root Cause — 0: Masalah yang disasar belum merupakan root cause (masih bisa dipertanyakan "kenapa" lagi) | 1: Sebagian masalah yang disasar sudah merupakan root cause final | 2: Semua masalah yang disasar sudah merupakan root cause final (tidak bisa dipertanyakan "kenapa" lagi)

TAHAP PLAN — 4. Tetapkan Perbaikan
11. Action Plan & PIC — 0: Tidak ada action plan dan PIC | 1: Sebagian action plan dan PIC ada | 2: Semua action plan dan PIC ada
12. Rencana Perbaikan per sumber masalah — 0: Tidak ada rencana perbaikan | 1: Sebagian masalah ada rencana perbaikan | 2: Semua sumber masalah punya rencana perbaikan yang jelas dan/atau prioritas penyelesaian yang baik
13. Form Usulan Perbaikan (FUP) — 0: Tidak ada pendaftaran FUP | 3: Sudah didaftarkan FUP namun belum dapat approval | 5: Sudah didaftarkan FUP dan sudah dapat approval

TAHAP DO — 5. Implementasi Perbaikan
14. Pelaksanaan Action Plan — 0: Tidak terlaksana semua | 1: Sebagian terlaksana | 2: Terlaksana semua
15. Dokumentasi Pelaksanaan — 0: Tidak ada bentuk dokumentasi kegiatan | 5: Sebagian kegiatan pelaksanaan sudah terdokumentasi (dokumen/report trial/visual) | 8: Semua kegiatan pelaksanaan sudah terdokumentasi

TAHAP CHECK — 6. Cek & Monitor Hasil
16. Pencapaian Target — 0: Tidak dapat menghubungkan antara hasil dengan target | 1: Dapat menghubungkan antara hasil dengan target
   WAJIB BANDINGKAN ANGKA SECARA EKSPLISIT: ambil angka TARGET yang disebutkan di kriteria 5 (Target SMART) pada awal dokumen, lalu bandingkan dengan angka HASIL AKHIR/pencapaian yang dilaporkan di bagian akhir dokumen. Skor 1 hanya diberikan jika ada keterhubungan yang jelas dan konsisten antara target awal dan hasil akhir — kalau target bergeser/berbeda dari rencana awal tanpa penjelasan yang jelas, itu HARUS disebutkan dan skor diturunkan.
17. Pengecekan Hasil — 0: Tidak dilakukan pengecekan hasil | 3: Dilakukan pengecekan hasil namun belum valid/tidak ada bukti | 5: Pengecekan hasil dilakukan dan valid, dibuktikan dengan evidence

TAHAP ACT — 9. Standardisasi
18. Kelengkapan Standardisasi (IK/SOP/OPL/Task CILT & PM/Centerline) — 0: Belum semua terstandarkan | 3: Sebagian sudah terstandarkan | 5: Sudah semua terstandarkan
19. Validasi Standardisasi — 0: Standar terbaru belum tervalidasi/disahkan oleh Sec Head Area | 1: Sebagian standar terbaru sudah terverifikasi & tervalidasi, dibuktikan dengan no register dokumen dan approval Sec Head Area | 2: Semua standar terbaru sudah terverifikasi & tervalidasi, dibuktikan dengan no register dokumen dan approval Sec Head Area
20. Tindak Lanjut Sosialisasi — 0: Belum sosialisasi | 3: Sebagian dibuktikan dengan dokumen sosialisasi (absensi) | 5: Dibuktikan lengkap dengan dokumen sosialisasi (absensi)
21. Replikasi ke area/mesin lain — 0: Area/mesin belum direplikasi | 3: Sebagian area/mesin sudah direplikasi ke area yang aplikatif | 5: Semua area/mesin sudah direplikasi ke area yang aplikatif, ATAU improvement memang tidak bisa direplikasi
"""

KATEGORI_IMPACT_14 = [
    "Gas / Steam", "Material Balance", "Manpower", "Downtime", "Waktu / Proses Kerja",
    "Overtime", "Listrik", "Air", "Stock Accuracy", "Inventory / Material Value",
    "DOI", "Quality", "Safety & Environment", "SOC & HTA",
]

# ==========================================
# 4. SESSION STATE
# ==========================================
_DEFAULTS = {
    "proses_selesai": False,
    "df_verifikasi": pd.DataFrame(),
    "df_alur_gemini": pd.DataFrame(), "df_alur_groq": pd.DataFrame(),
    "df_rubrik_gemini": pd.DataFrame(), "df_rubrik_groq": pd.DataFrame(),
    "df_banding": pd.DataFrame(),
    "total_skor_gemini": 0.0, "total_skor_groq": 0.0,
    "df_saving_gemini": pd.DataFrame(), "df_saving_groq": pd.DataFrame(),
    "df_feedback_gemini": pd.DataFrame(), "df_feedback_groq": pd.DataFrame(),
    "transkrip": [],
    "log_error": [],
    "konteks_groq": {},
    "df_alur_banding": pd.DataFrame(), "df_saving_banding": pd.DataFrame(), "df_feedback_banding": pd.DataFrame(),
    "nama_file": "Dokumen_Kaizen",
}
for _k, _val in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _val.copy() if hasattr(_val, "copy") else _val

# ==========================================
# 5. MESIN AI & PARSER
# ==========================================
class NullLog:
    """Pengganti log UI untuk thread paralel (jangan panggil fungsi Streamlit dari thread)."""
    def write(self, *args, **kwargs):
        pass


NULL = NullLog()


class BufferLog:
    """Menampung pesan log dari thread; thread utama mengambilnya berkala untuk ditampilkan langsung."""
    def __init__(self):
        self._pesan = []
        self._kunci = threading.Lock()

    def write(self, teks, *args, **kwargs):
        with self._kunci:
            self._pesan.append(str(teks))

    def ambil(self):
        with self._kunci:
            baru = self._pesan[:]
            self._pesan.clear()
        return baru


def ringkas(teks, batas):
    """Potong teks panjang (dipakai agar prompt Groq muat di limit token)."""
    teks = teks or ""
    if len(teks) <= batas:
        return teks
    return teks[:batas] + "\n...[dipotong agar muat limit token]"


def _tunggu_dari_pesan(pesan, default):
    m = re.search(r"try again in (?:(\d+)m)?\s*([\d.]+)s", pesan, flags=re.IGNORECASE)
    if m:
        menit = int(m.group(1) or 0)
        return min(menit * 60 + float(m.group(2)) + 1, 90)
    return default


def panggil_gemini(contents, deskripsi, log_ui, config=None, maksimal_percobaan=3):
    config = config or CONFIG_TEXT
    for _ in range(maksimal_percobaan):
        try:
            log_ui.write(f"⏳ **{deskripsi} (Gemini):** sedang menganalisis...")
            response = client_gemini.models.generate_content(model=MODEL_GEMINI, contents=contents, config=config)
            teks = response.text if response and response.text else ""
            if not teks:
                time.sleep(8)
                continue
            log_ui.write(f"✅ **{deskripsi} (Gemini):** selesai.")
            time.sleep(COOLDOWN_GEMINI)
            return teks
        except Exception as e:
            pesan = str(e)
            if "429" in pesan or "RESOURCE_EXHAUSTED" in pesan.upper():
                tunggu = _tunggu_dari_pesan(pesan, 65)
                log_ui.write(f"⚠️ **Gemini:** limit rate. Menunggu {tunggu:.0f} detik...")
                time.sleep(tunggu)
            else:
                log_ui.write(f"⚠️ **Gemini:** error: {e}. Mencoba ulang...")
                time.sleep(10)
    log_ui.write(f"❌ **{deskripsi} (Gemini):** gagal setelah {maksimal_percobaan} percobaan.")
    return ""


def estimasi_token(teks, max_output):
    return int(len(teks) / GROQ_CHAR_PER_TOKEN) + max_output + 120


def panggil_groq(prompt_text, deskripsi, log_ui, maksimal_percobaan=3, max_output=None):
    max_output = max_output or GROQ_MAX_OUTPUT
    if PACER.habis():
        log_ui.write(f"❌ **{deskripsi} (Groq):** dilewati — kuota token HARIAN Groq sudah habis.")
        return ""
    estimasi = estimasi_token(prompt_text, max_output)
    if estimasi > GROQ_TPM:
        log_ui.write(
            f"⚠️ **{deskripsi} (Groq):** estimasi {estimasi:,} token melebihi GROQ_TPM ({GROQ_TPM:,}); berisiko ditolak (413)."
        )
    for _ in range(maksimal_percobaan):
        entri = PACER.tunggu(estimasi, log_ui)
        try:
            log_ui.write(f"⏳ **{deskripsi} (Groq):** sedang mengevaluasi (±{estimasi:,} token)...")
            response = client_groq.chat.completions.create(
                model=MODEL_GROQ,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Anda adalah asisten auditor Kaizen tingkat senior. ANDA WAJIB MENGELUARKAN OUTPUT DALAM "
                            "BENTUK JSON ARRAY SAJA (dimulai dengan [ dan diakhiri dengan ]). DILARANG KERAS menambah "
                            "teks pengantar, penutup, atau tanda markdown. Hanya JSON murni."
                        ),
                    },
                    {"role": "user", "content": prompt_text},
                ],
                temperature=0.2,
                max_completion_tokens=max_output,
                **GROQ_EXTRA,
            )
            pilihan = response.choices[0]
            teks = pilihan.message.content or ""
            pakai = getattr(getattr(response, "usage", None), "total_tokens", None) or estimasi
            PACER.catat(entri, pakai)
            if pilihan.finish_reason == "length":
                log_ui.write(
                    f"⚠️ **{deskripsi} (Groq):** output terpotong (finish_reason=length). "
                    "Naikkan GROQ_MAX_OUTPUT atau turunkan GROQ_REASONING."
                )
            if not teks.strip():
                log_ui.write(
                    f"⚠️ **{deskripsi} (Groq):** respons kosong (biasanya token habis dipakai reasoning). Mencoba ulang..."
                )
                time.sleep(3)
                continue
            log_ui.write(f"✅ **{deskripsi} (Groq):** selesai ({pakai:,} token terpakai).")
            return teks
        except Exception as e:
            pesan = str(e)
            pesan_kecil = pesan.lower()
            if "413" in pesan or "request too large" in pesan_kecil:
                PACER.batal(entri)
                log_ui.write(
                    f"❌ **{deskripsi} (Groq):** request terlalu besar untuk limit token (413). "
                    "Turunkan GROQ_TPM/GROQ_MAX_OUTPUT atau perkecil GROQ_CHAR_PER_TOKEN agar konteks dipotong lebih banyak."
                )
                return ""
            if "429" in pesan:
                if "per day" in pesan_kecil:
                    PACER.tandai_habis()
                    log_ui.write(f"❌ **{deskripsi} (Groq):** kuota harian Groq habis (TPD/RPD). Coba lagi besok.")
                    return ""
                tunggu = _tunggu_dari_pesan(pesan, 15)
                log_ui.write(f"⚠️ **Groq:** limit per menit tercapai. Menunggu {tunggu:.0f} detik...")
                time.sleep(tunggu)
            else:
                PACER.batal(entri)
                log_ui.write(f"⚠️ **Groq:** error: {e}. Mencoba ulang...")
                time.sleep(5)
    log_ui.write(f"❌ **{deskripsi} (Groq):** gagal setelah {maksimal_percobaan} percobaan.")
    return ""


def paralel(log, catatan, kerja_gemini, kerja_groq):
    """Jalankan tugas Gemini & Groq bersamaan. Log dari thread ditampilkan langsung oleh thread utama."""
    bg, bq = BufferLog(), BufferLog()

    def kuras():
        for lg in (bg, bq):
            for pesan in lg.ambil():
                log.write(pesan)
                if "❌" in pesan or "⚠️" in pesan:
                    catatan.append(pesan)

    with ThreadPoolExecutor(max_workers=2) as ex:
        a = ex.submit(kerja_gemini, bg)
        b = ex.submit(kerja_groq, bq)
        while not (a.done() and b.done()):
            kuras()
            time.sleep(1)
        kuras()
        return a.result(), b.result()


def bersihkan_dan_parse_json(teks_raw):
    """Parser JSON toleran: buang markdown, cari array/objek, fallback ke parse langsung."""
    if not teks_raw:
        return []
    teks = re.sub(r"```(?:json)?", "", teks_raw, flags=re.IGNORECASE).strip()
    kandidat = [teks]
    m = re.search(r"\[.*\]", teks, re.DOTALL)
    if m:
        kandidat.append(m.group(0))
    m = re.search(r"\{.*\}", teks, re.DOTALL)
    if m:
        kandidat.append(m.group(0))
    for k in kandidat:
        try:
            data = json.loads(k)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return [d for d in v if isinstance(d, dict)]
            return [data]
    # Fallback terakhir: JSON rusak/terpotong → selamatkan objek-objek yang utuh
    dec = json.JSONDecoder()
    i, objek = 0, []
    while True:
        i = teks.find("{", i)
        if i == -1:
            break
        try:
            obj, j = dec.raw_decode(teks, i)
            if isinstance(obj, dict):
                objek.append(obj)
            i = j
        except json.JSONDecodeError:
            i += 1
    return objek


def bersihkan_sel(df):
    """Ubah sel berisi list/dict jadi teks agar aman untuk tabel & Excel."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
    return df


def buat_df(data_json):
    return bersihkan_sel(pd.DataFrame(data_json)) if data_json else pd.DataFrame()


def _ke_int(x, default=0):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def _ke_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def tentukan_validasi_manual(item, nomor_kriteria):
    nilai_mentah = item.get("perlu_validasi_manual")
    if nilai_mentah is None:
        perlu = nomor_kriteria in KRITERIA_RUJUKAN_VALIDASI_MANUAL
        return perlu, (KRITERIA_RUJUKAN_VALIDASI_MANUAL.get(nomor_kriteria, "") if perlu else "")
    perlu = str(nilai_mentah).strip().upper() in ("YA", "TRUE", "1", "YES")
    alasan = str(item.get("alasan_validasi_manual", "")).strip() if perlu else ""
    return perlu, alasan


KOLOM_RUBRIK = ["no", "kriteria", "status validasi", "alasan_manual", "skor", "cek_skor", "justifikasi", "perlu_manual"]


def format_tabel_rubrik(json_data):
    """Rapikan output rubrik 1 model → DataFrame standar + total skor."""
    baris = {}
    for item in json_data:
        no = _ke_int(item.get("no", item.get("No", item.get("nomor"))))
        if no not in RUBRIK_META or no in baris:
            continue
        skor = _ke_float(item.get("skor", item.get("score", item.get("nilai"))))
        perlu, alasan = tentukan_validasi_manual(item, no)
        cek = ""
        if skor is None:
            cek = "⚠️ skor kosong/bukan angka"
        elif int(skor) != skor or int(skor) not in RUBRIK_META[no][2]:
            cek = f"⚠️ di luar pilihan rubrik {sorted(RUBRIK_META[no][2])}"
        baris[no] = {
            "no": no,
            "kriteria": RUBRIK_META[no][1],
            "status validasi": "⚠️ VALIDASI MANUAL" if perlu else "OTOMATIS AI",
            "alasan_manual": alasan,
            "skor": skor,
            "cek_skor": cek,
            "justifikasi": str(item.get("justifikasi", item.get("alasan", item.get("keterangan", "")))),
            "perlu_manual": perlu,
        }
    df = pd.DataFrame([baris[n] for n in sorted(baris)], columns=KOLOM_RUBRIK)
    total = float(pd.to_numeric(df["skor"], errors="coerce").sum()) if not df.empty else 0.0
    return df, total


def gabungkan_rubrik(df_gem, df_groq):
    """Gabungkan skor Gemini & Groq jadi SATU tabel berdampingan (kunci gabung: nomor kriteria)."""
    def siapkan(df, sfx):
        sub = pd.DataFrame(columns=KOLOM_RUBRIK) if (df is None or df.empty) else df[KOLOM_RUBRIK].copy()
        sub = sub[["no", "skor", "justifikasi", "perlu_manual", "alasan_manual"]].astype({"no": "int64"})
        return sub.rename(columns={c: f"{c}_{sfx}" for c in sub.columns if c != "no"})

    base = pd.DataFrame({"no": list(RUBRIK_META.keys())})
    m = base.merge(siapkan(df_gem, "gem"), on="no", how="left").merge(siapkan(df_groq, "groq"), on="no", how="left")
    m["skor_gem"] = pd.to_numeric(m["skor_gem"], errors="coerce")
    m["skor_groq"] = pd.to_numeric(m["skor_groq"], errors="coerce")

    baris = []
    for _, r in m.iterrows():
        no = int(r["no"])
        ada_dua = pd.notna(r["skor_gem"]) and pd.notna(r["skor_groq"])
        sama = ada_dua and r["skor_gem"] == r["skor_groq"]
        beda = ada_dua and not sama
        if not ada_dua:
            hasil = "❓ Data tidak lengkap"
        else:
            hasil = "✅ Sama" if sama else "⚠️ Beda"

        man_gem = bool(r["perlu_manual_gem"]) if pd.notna(r["perlu_manual_gem"]) else False
        man_groq = bool(r["perlu_manual_groq"]) if pd.notna(r["perlu_manual_groq"]) else False
        alasan = []
        if man_gem and pd.notna(r["alasan_manual_gem"]) and str(r["alasan_manual_gem"]).strip():
            alasan.append(f"Gemini: {r['alasan_manual_gem']}")
        if man_groq and pd.notna(r["alasan_manual_groq"]) and str(r["alasan_manual_groq"]).strip():
            alasan.append(f"Groq: {r['alasan_manual_groq']}")
        if beda:
            alasan.append(f"Skor berbeda (Gemini {r['skor_gem']:g} vs Groq {r['skor_groq']:g})")
        perlu_manual = man_gem or man_groq or beda

        baris.append({
            "No": no,
            "Tahap": RUBRIK_META[no][0],
            "Kriteria": RUBRIK_META[no][1],
            "Skor Gemini": r["skor_gem"],
            "Skor Groq": r["skor_groq"],
            "Hasil Banding": hasil,
            "Status Validasi": "⚠️ VALIDASI MANUAL" if perlu_manual else "OTOMATIS AI",
            "Alasan Validasi Manual": " | ".join(alasan),
            # Otomatis terisi hanya jika kedua AI sepakat; kalau beda, juri yang memutuskan.
            "Skor Final (Juri)": r["skor_gem"] if sama else None,
            "Catatan Validator": "",
            "Justifikasi Gemini": "" if pd.isna(r["justifikasi_gem"]) else r["justifikasi_gem"],
            "Justifikasi Groq": "" if pd.isna(r["justifikasi_groq"]) else r["justifikasi_groq"],
        })
    df = pd.DataFrame(baris)
    df["Skor Final (Juri)"] = pd.to_numeric(df["Skor Final (Juri)"], errors="coerce")
    return df


# ==========================================
# 5b. PENGGABUNGAN HASIL GEMINI vs GROQ (1 TABEL PER TOPIK)
# ==========================================
ALUR_KODE = (
    [f"P{i}" for i in range(1, 8)] + [f"D{i}" for i in range(1, 5)]
    + [f"C{i}" for i in range(1, 3)] + [f"A{i}" for i in range(1, 5)]
)
FEEDBACK_KATEGORI = [
    "Struktur & Kejelasan Penulisan",
    "Perumusan Problem Statement",
    "Kesesuaian Goal/Target dengan Objective Awal",
    "Kedalaman Analisis Akar Masalah",
    "Kekuatan Bukti & Data Pendukung",
    "Standardisasi & Keberlanjutan",
]


def _baris_dict(df):
    return [] if df is None or df.empty else df.to_dict("records")


def _alias(row, *nama, default=""):
    for n in nama:
        v = row.get(n)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if str(v).strip() != "":
            return v
    return default


def _norm_kunci(x):
    return re.sub(r"[\s/_\-&]+", "", str(x or "").lower())


def _ke_kanon(nama, kanon):
    """Petakan nama kategori dari model ke nama baku (toleran beda spasi/ejaan kecil). None jika tak cocok."""
    n = _norm_kunci(nama)
    if not n:
        return None
    peta = {_norm_kunci(k): k for k in kanon}
    if n in peta:
        return peta[n]
    cocok = difflib.get_close_matches(n, list(peta), n=1, cutoff=0.75)
    return peta[cocok[0]] if cocok else None


def _hasil_banding(a, b):
    a, b = str(a or "").strip().upper(), str(b or "").strip().upper()
    if not a or not b:
        return "❓ Data tidak lengkap"
    return "✅ Sama" if a == b else "⚠️ Beda"


def gabungkan_alur(df_gem, df_groq):
    peta = {"Gemini": {}, "Groq": {}}
    for nama, df in (("Gemini", df_gem), ("Groq", df_groq)):
        for r in _baris_dict(df):
            k = str(_alias(r, "no", "No", "kode")).strip().upper()
            if k and k not in peta[nama]:
                peta[nama][k] = r
    semua = set(peta["Gemini"]) | set(peta["Groq"])
    kunci = [k for k in ALUR_KODE if k in semua] + sorted(k for k in semua if k not in ALUR_KODE)
    baris = []
    for k in kunci:
        rg, rq = peta["Gemini"].get(k, {}), peta["Groq"].get(k, {})
        vg = str(_alias(rg, "verdict", "status")).strip().upper()
        vq = str(_alias(rq, "verdict", "status")).strip().upper()
        baris.append({
            "No": k,
            "Fase": _alias(rg, "fase", default=_alias(rq, "fase")),
            "Tahap": _alias(rg, "tahap", default=_alias(rq, "tahap")),
            "Verdict Gemini": vg,
            "Verdict Groq": vq,
            "Hasil Banding": _hasil_banding(vg, vq),
            "Temuan Gemini": _alias(rg, "temuan", "keterangan"),
            "Temuan Groq": _alias(rq, "temuan", "keterangan"),
        })
    kolom = ["No", "Fase", "Tahap", "Verdict Gemini", "Verdict Groq", "Hasil Banding", "Temuan Gemini", "Temuan Groq"]
    return bersihkan_sel(pd.DataFrame(baris, columns=kolom))


def gabungkan_saving(df_gem, df_groq):
    data = {"Gemini": {}, "Groq": {}}
    ekstra = {}  # kunci ternormalisasi -> nama tampil (mis. "Jenis Saving")
    for nama, df in (("Gemini", df_gem), ("Groq", df_groq)):
        for r in _baris_dict(df):
            kat = str(_alias(r, "kategori", "Kategori")).strip()
            if not kat:
                continue
            baku = _ke_kanon(kat, KATEGORI_IMPACT_14) or ekstra.setdefault(_norm_kunci(kat), kat)
            data[nama].setdefault(baku, r)
    semua = set(data["Gemini"]) | set(data["Groq"])
    kunci = [k for k in KATEGORI_IMPACT_14 if k in semua] + [v for v in ekstra.values() if v in semua]
    baris = []
    for k in kunci:
        rg, rq = data["Gemini"].get(k, {}), data["Groq"].get(k, {})
        sg = str(_alias(rg, "status")).strip()
        sq = str(_alias(rq, "status")).strip()
        baris.append({
            "Kategori": k,
            "Status Gemini": sg,
            "Status Groq": sq,
            "Hasil Banding": _hasil_banding(sg, sq),
            "Keterangan Gemini": _alias(rg, "keterangan", "alasan", "catatan"),
            "Keterangan Groq": _alias(rq, "keterangan", "alasan", "catatan"),
        })
    kolom = ["Kategori", "Status Gemini", "Status Groq", "Hasil Banding", "Keterangan Gemini", "Keterangan Groq"]
    return bersihkan_sel(pd.DataFrame(baris, columns=kolom))


def gabungkan_feedback(df_gem, df_groq):
    data = {"Gemini": {}, "Groq": {}}
    for nama, df in (("Gemini", df_gem), ("Groq", df_groq)):
        n_ekstra = 0
        for r in _baris_dict(df):
            kat = str(_alias(r, "kategori", "Kategori")).strip()
            baku = _ke_kanon(kat, FEEDBACK_KATEGORI)
            if baku is None:
                n_ekstra += 1
                baku = f"Catatan Tambahan {n_ekstra}"
            data[nama].setdefault(baku, r)
    semua = set(data["Gemini"]) | set(data["Groq"])
    kunci = [k for k in FEEDBACK_KATEGORI if k in semua] + sorted(k for k in semua if k not in FEEDBACK_KATEGORI)
    baris = []
    for k in kunci:
        rg, rq = data["Gemini"].get(k, {}), data["Groq"].get(k, {})
        baris.append({
            "Kategori": k,
            "Kekuatan (Gemini)": _alias(rg, "kekuatan"),
            "Kekuatan (Groq)": _alias(rq, "kekuatan"),
            "Area Perbaikan (Gemini)": _alias(rg, "area_perbaikan", "area perbaikan"),
            "Area Perbaikan (Groq)": _alias(rq, "area_perbaikan", "area perbaikan"),
            "Saran Konkret (Gemini)": _alias(rg, "saran_konkret", "saran konkret", "saran"),
            "Saran Konkret (Groq)": _alias(rq, "saran_konkret", "saran konkret", "saran"),
        })
    kolom = [
        "Kategori", "Kekuatan (Gemini)", "Kekuatan (Groq)", "Area Perbaikan (Gemini)", "Area Perbaikan (Groq)",
        "Saran Konkret (Gemini)", "Saran Konkret (Groq)",
    ]
    return bersihkan_sel(pd.DataFrame(baris, columns=kolom))


# ==========================================
# 6. PROMPT
# ==========================================
def prompt_ekstraksi():
    return (
        "Anda berperan sebagai Analis Ekstraksi Bukti Dokumen Kaizen yang teliti dan hanya melaporkan fakta yang benar-benar tertulis/tervisualisasi di dokumen, tanpa mengarang, karena akan dipakai untuk analisis koherensi logika, bukan sekadar cek ada/tidak.\n\n"
        "ATURAN PENTING #1 — CARI MAKNA TERSURAT DAN TERSIRAT: banyak dokumen TIDAK menuliskan masalah/target/goal/hasil-saving dengan label eksplisit yang jelas (misal tidak ada section 'Target:' secara langsung), tapi maknanya tersirat di kalimat lain (misal disebutkan sekilas dalam narasi solusi atau kesimpulan). Untuk SETIAP elemen di poin 1, 2, dan 7 di bawah: kalau tidak ada label eksplisit, telusuri SELURUH dokumen untuk kalimat yang secara implisit mengandung makna elemen itu. Kutip kalimat aslinya, sebutkan halaman berapa, DAN tandai dengan jelas apakah itu 'EKSPLISIT' (ada label jelas) atau 'IMPLISIT/TERSIRAT' (disimpulkan dari kalimat lain, sebutkan dari kalimat mana).\n\n"
        "ATURAN PENTING #2 — JANGAN CUMA SEBUT HALAMAN: setiap kali merujuk suatu halaman/bagian dokumen, WAJIB jelaskan APA ISI KONKRETNYA dan APA ANGKA/MEASUREMENT-nya di situ. DILARANG menulis rujukan kosong seperti 'ada di halaman 24 dan 31' tanpa penjelasan — itu tidak berguna untuk penilaian yang butuh angka pengukuran jelas sebagai faktor penentu.\n\n"
        "ATURAN PENTING #3 — BEDAKAN TANGGAL HEADER DOKUMEN KONTROL DENGAN TANGGAL AKTUAL KEGIATAN: banyak formulir perusahaan (One Point Lesson, Daftar Hadir, IK/SOP, dsb.) memiliki header 'document control' berisi field seperti 'No. Dokumen', 'Tanggal Berlaku', 'Revisi', dan 'Halaman'. Field-field ini menjelaskan STATUS TEMPLATE/FORMULIR itu sendiri (kapan versi form tersebut disahkan untuk dipakai secara umum di perusahaan), BUKAN tanggal kejadian/aktivitas spesifik yang dicatat memakai formulir itu. JANGAN PERNAH melaporkan 'Tanggal Berlaku' pada header dokumen kontrol sebagai anomali/inkonsistensi timeline proyek — itu bukan indikasi kesalahan dan bukan pembanding yang valid. Tanggal AKTUAL kegiatan biasanya berada di badan formulir (misal field 'Tanggal/Jam Pelaksanaan', tanggal tulisan tangan pada baris data, dsb.), bukan di header dokumen. Untuk SETIAP tanggal yang diekstrak dari sebuah formulir, WAJIB sebutkan secara eksplisit sumbernya: 'tanggal berlaku template (header dokumen kontrol)' atau 'tanggal aktual pelaksanaan (isi formulir)'. HANYA tanggal aktual pelaksanaan yang relevan dibandingkan dengan timeline proyek.\n\n"
        "ATURAN PENTING #4 — MEMBACA TABEL IMPACT/MANFAAT DENGAN AMBANG BATAS SKALA: kalau dokumen memuat tabel dengan format kategori impact diikuti 2 kolom ambang batas/skala penilaian (misal 'Mengurangi ≤ 1%' vs 'Mengurangi >5%') dan 1 kolom penjelasan/keterangan di ujung kanan, kolom ambang batas itu BUKAN bukti pencapaian aktual — HANYA kolom penjelasan/keterangan yang berisi pencapaian aktual proyek ini. Kalau kolom penjelasan untuk suatu kategori KOSONG, laporkan kategori itu sebagai 'TIDAK ADA PENJELASAN/PENCAPAIAN YANG DILAPORKAN', JANGAN mengarang angka dari kolom ambang batas.\n\n"
        "Ekstrak SETIAP elemen berikut secara VERBATIM/detail (kutip isi aslinya, jangan diringkas berlebihan):\n\n"
        "1. MASALAH UTAMA: kondisi awal, DATA KUANTITATIF pendukung (sebutkan angka, satuan, DAN periode/metode pengukurannya persis, misal 'rata-rata 3 bulan Jan-Mar 2024'), 5W1H lengkap (What/Where/When/Who/Why/How), evidence 5G.\n"
        "2. TARGET AWAL (SMART): kutip persis angka/kalimat target yang ditetapkan di awal dokumen (eksplisit ATAU implisit sesuai Aturan #1).\n"
        "3. FISHBONE DIAGRAM: untuk SETIAP cabang/duri yang ada, sebutkan (a) kategori 4M yang dipakai dokumen (Man/Method/Machine/Material), (b) isi penyebab yang dituliskan di cabang itu. Buat sebagai daftar, contoh: 'Man: operator kurang terlatih', 'Machine: mesin sering aus'.\n"
        "4. ANALISIS 5 WHYS: kutip SETIAP baris why secara berurutan dan lengkap (why 1 sampai why terakhir) apa adanya, jangan diringkas. Sebutkan juga apa root cause final yang diklaim dokumen.\n"
        "5. ACTION PLAN & PIC: daftar rencana perbaikan beserta penanggung jawab (PIC) BESERTA JABATAN/PERANNYA bila disebutkan (misal 'Budi - Teknisi Mesin'), dan status FUP (Form Usulan Perbaikan) bila disebutkan.\n"
        "6. IMPLEMENTASI: bukti pelaksanaan (dokumentasi, foto before/after, laporan trial).\n"
        "7. HASIL AKHIR/PENCAPAIAN (termasuk SAVING): kutip persis angka hasil akhir yang dilaporkan, SATUAN, dan periode/metode pengukurannya persis (untuk dibandingkan dengan metode pengukuran kondisi awal di poin 1) — eksplisit ATAU implisit sesuai Aturan #1. Ikuti Aturan #4 kalau data ini berasal dari tabel impact bertingkat ambang batas.\n"
        "8. STANDARDISASI: dokumen IK/SOP/OPL/CILT/PM/Centerline yang dibuat, status validasi/approval, bukti sosialisasi (absensi — sebutkan SIAPA/JABATAN APA yang mengikuti bila ada), dan bukti replikasi ke area/mesin lain (sebutkan karakteristik area tujuan replikasi bila disebutkan, untuk menilai apakah memang sejenis/sepadan dengan area asal masalah). Untuk setiap tanggal yang ditemukan di dokumen standardisasi/sosialisasi ini, WAJIB ikuti Aturan #3 di atas — bedakan tanggal berlaku template dengan tanggal aktual pelaksanaan sebelum menyimpulkan apa pun.\n"
        "9. KUALITAS PENULISAN: catat kalau ada typo/salah ketik yang cukup mengganggu, kalimat ambigu/membingungkan, atau bagian yang tidak konsisten penomoran/formatnya (untuk bahan feedback ke peserta, bukan untuk skor rubrik). JANGAN memasukkan tanggal berlaku template dokumen kontrol (Aturan #3) sebagai contoh kesalahan penulisan di sini.\n\n"
        "Jika suatu elemen tidak ditemukan di dokumen sama sekali (baik eksplisit maupun implisit), nyatakan dengan jelas 'TIDAK DITEMUKAN' — jangan mengarang."
    )


def prompt_verifikasi():
    return (
        "Anda adalah Analis Verifikasi Bukti Visual & Kelayakan (senior) yang SANGAT KRITIS terhadap kualitas dasar submission Kaizen. Banyak peserta kompetisi ini belum paham konsep PDCA dengan baik, dan beberapa submission bahkan BUKAN merupakan proyek improvement sama sekali (misal: cuma laporan rutin, pengadaan barang tanpa problem-solving, atau aktivitas maintenance biasa yang dibungkus format Kaizen). Tugas Anda membongkar ini dengan membaca LANGSUNG dokumen PDF (termasuk semua foto/gambar/diagram di dalamnya), bukan cuma ringkasan.\n\n"
        "ATURAN WAJIB:\n"
        "1. Di kolom 'catatan', JANGAN cuma menyebut nomor halaman — selalu jelaskan ISI KONKRET apa yang ada di situ (dan angka/measurement-nya kalau relevan). Pertimbangkan juga bahwa target/masalah/hasil kadang tertulis IMPLISIT/tersirat di kalimat lain, bukan cuma yang berlabel eksplisit — telusuri keduanya.\n"
        "2. BEDAKAN tanggal berlaku TEMPLATE formulir (header document control: field 'No. Dokumen', 'Tanggal Berlaku', 'Revisi', 'Halaman') dengan tanggal AKTUAL pelaksanaan kegiatan (biasanya di badan formulir, misal field 'Tanggal/Jam Pelaksanaan' atau tanggal tulisan tangan pada baris data). JANGAN melaporkan tanggal berlaku template sebagai anomali/inkonsistensi dibanding timeline proyek — itu bukan pembanding yang valid. Kalau melaporkan temuan terkait tanggal, sebutkan eksplisit jenis tanggalnya (tanggal berlaku template ATAU tanggal aktual pelaksanaan).\n\n"
        "Lakukan 4 pemeriksaan berikut:\n\n"
        "## A. GATE CHECK — Kelayakan sebagai Proyek Improvement\n"
        "Apakah dokumen ini benar-benar proyek continuous improvement yang valid? Tanda-tanda TIDAK LAYAK: tidak ada kondisi awal/masalah yang didefinisikan dengan jelas, tidak ada perubahan before-after yang nyata, tidak ada analisis akar masalah sama sekali (langsung lompat ke solusi), atau isinya sebenarnya laporan administratif/aktivitas rutin yang dipaksakan ke format Kaizen. Beri verdict: 'LAYAK' (jelas proyek improvement yang sah), 'PERLU PERHATIAN' (ada keraguan, perlu ditinjau juri), atau 'TIDAK LAYAK' (bukan proyek improvement).\n\n"
        "## B. KEBENARAN SEMANTIK 5W1H\n"
        "Untuk MASING-MASING elemen (What, Where, When, Who, Why, How — atau elemen serupa yang dipakai dokumen), periksa apakah ISI yang dituliskan benar-benar menjawab pertanyaan elemen itu, bukan cuma ada teks di kolomnya. Contoh kesalahan yang harus ditangkap: isi kolom 'How' sebenarnya menjelaskan 'Where' (lokasi), atau isi 'Which'/kolom lain tertukar dengan elemen lain. Tandai tiap elemen SESUAI atau TERTUKAR/TIDAK SESUAI dengan penjelasan spesifik.\n\n"
        "## C. AUDIT FOTO & BUKTI VISUAL (ANTI-MANIPULASI)\n"
        "Untuk SETIAP foto/gambar/diagram penting yang kamu lihat di dokumen (terutama foto before/after, dan diagram fishbone/flow), deskripsikan singkat apa yang benar-benar terlihat di foto itu. \n"
        "KRITIKAL: Cek dengan sangat teliti apakah foto 'Before' dan 'After' sebenarnya adalah foto yang sama persis namun hanya diubah sudut pandangnya (angle), di-zoom, atau di-crop tanpa ada perubahan fisik yang nyata! Tandai SESUAI kalau foto benar-benar menunjukkan perbaikan/perubahan nyata sesuai klaim, atau MERAGUKAN kalau foto before/after terlihat identik (indikasi rekayasa), tidak relevan, atau tampak diambil dari konteks lain.\n\n"
        "## D. AUDIT KELENGKAPAN FORM USULAN PERBAIKAN (FUP)\n"
        "Cari secara spesifik dokumen/halaman yang diklaim sebagai Form Usulan Perbaikan (FUP). Dokumen FUP yang sah HARUS memenuhi syarat visual berikut:\n"
        "1. Memiliki Kop Surat perusahaan resmi.\n"
        "2. Terdapat judul/keyword 'Form Usulan Perbaikan' atau 'FUP'.\n"
        "3. Terdapat kolom tanda tangan persetujuan (Approval) yang SUDAH DITANDATANGANI.\n"
        "JANGAN menganggap form standardisasi (OPL/IK/SOP) atau daftar hadir sosialisasi sebagai FUP. Tandai status sebagai 'ADA DAN APPROVED' (jika ada form FUP dan sudah di-acc), 'ADA TAPI BELUM APPROVED' (jika ada form FUP tapi kolom tanda tangan kosong/belum lengkap), atau 'TIDAK DITEMUKAN / SALAH DOKUMEN' (jika yang dilampirkan adalah dokumen lain seperti OPL/SOP atau tidak ada sama sekali).\n\n"
        "Keluarkan HANYA JSON array valid (satu array datar berisi semua temuan A+B+C+D, dibedakan lewat field 'kategori'), dengan skema persis:\n"
        '[{"kategori": "GATE CHECK", "item": "Kelayakan Proyek Improvement", "status": "LAYAK", "catatan": "alasan spesifik merujuk isi dokumen"}, {"kategori": "5W1H", "item": "How", "status": "TERTUKAR/TIDAK SESUAI", "catatan": "isi kolom How sebenarnya menjelaskan lokasi (Where), bukan metode"}, {"kategori": "FOTO", "item": "Foto halaman 8 (before)", "status": "MERAGUKAN", "catatan": "foto before dan after terlihat seperti foto yang sama hanya di-zoom"}, {"kategori": "FUP", "item": "Form Usulan Perbaikan", "status": "TIDAK DITEMUKAN / SALAH DOKUMEN", "catatan": "yang dilampirkan adalah form OPL, bukan FUP resmi"}]'
    )


def prompt_alur(laporan_ekstraksi, raw_verifikasi):
    return f"""Anda adalah Analis Audit Konsistensi Metodologi PDCA (QC-Story) yang menelusuri "benang merah" (golden thread): apakah tiap tools di tiap fase PDCA benar-benar tersambung MASUK AKAL secara teknis/operasional ke tools sebelum dan sesudahnya — bukan cuma sama-sama ada di dokumen.

Gunakan pengetahuan umum troubleshooting industri sebagai patokan kewajaran sebab-akibat. Contoh MASUK AKAL: "mesin macet" -> kenapa? "bearing aus" -> kenapa? "kurang pelumasan" -> kenapa? "tidak ada jadwal preventive maintenance". Contoh TIDAK MASUK AKAL: "mesin macet" tiba-tiba dijawab "operator kurang training" tanpa penjelasan penghubung.

ATURAN WAJIB UNTUK SETIAP TEMUAN: jangan cuma menyebut nomor halaman — jelaskan ISI KONKRET dan ANGKA/MEASUREMENT yang ada di situ. Pertimbangkan juga bahwa sebagian data (target/masalah/hasil) mungkin tertulis IMPLISIT (lihat penanda EKSPLISIT/IMPLISIT dari hasil ekstraksi di bawah), bukan cuma yang berlabel jelas.

DATA HASIL EKSTRAKSI DOKUMEN:
{laporan_ekstraksi}

HASIL VERIFIKASI KEBENARAN 5W1H, FOTO & KELAYAKAN:
{raw_verifikasi}

Telusuri dan evaluasi SEMUA titik sambungan berikut, dikelompokkan per fase PDCA:

## FASE PLAN (P1-P7)
P1. "5G ke 5W1H": apakah bukti observasi lapangan (5G) konsisten dan mendukung detail yang dilaporkan di 5W1H, bukan cuma tempelan formalitas?
P2. "5W1H ke Problem Statement": apakah rangkuman masalah mencerminkan semua elemen 5W1H DENGAN BENAR (cek hasil verifikasi — kalau ada elemen TERTUKAR, ini otomatis TIDAK KONSISTEN), tidak ada yang hilang/melenceng?
P3. "Data/Losses ke Target SMART": apakah target yang ditetapkan punya justifikasi ANGKA yang jelas dari data losses/kondisi awal (misal target reduksi 30% harus bisa ditelusuri dari angka awal vs angka target), atau target itu muncul begitu saja tanpa perhitungan?
P4. "Problem Statement ke Kepala Ikan Fishbone": apakah efek/masalah utama di kepala ikan SAMA dengan problem statement, atau melenceng ke masalah lain?
P5. "Kategori 4M pada Fishbone": untuk SETIAP cabang, apakah penyebabnya masuk akal di kategori 4M itu? Sebutkan SPESIFIK cabang yang salah kategori kalau ada.
P6. "Cabang Fishbone ke Rantai Why-Why": apakah analisis why-why benar-benar berangkat dari salah satu cabang/penyebab di fishbone (bukan topik baru yang tiba-tiba muncul)? DAN untuk SETIAP pasangan why berurutan, apakah why berikutnya penyebab TEKNIS LANGSUNG dari why sebelumnya (bukan cuma berkaitan tema)? Sebutkan link why-ke-berapa yang lemah/meloncat kalau ada.
P7. "Root Cause Akhir ke Bukti Pendukung": KRITIKAL: Evaluasi apakah SETIAP root cause akhir benar-benar merupakan FAKTA yang BISA DIBUKTIKAN secara objektif di dokumen, atau hanya berupa ASUMSI/POTENSI/HIPOTESIS (misalnya: peserta mengklaim "terjadi reaksi kimia X" tapi tidak ada lampiran hasil uji lab, atau mengklaim "operator kelelahan" tanpa data beban kerja). Jika ada root cause yang bersifat spekulatif, asumtif, atau "masih potensi" tanpa pembuktian data/validasi teknis di dokumen, status P7 WAJIB "LEMAH". Sebutkan secara spesifik root cause mana yang hanya berupa tebakan/potensi.

## FASE DO (D1-D4)
D1. "Root Cause ke Action Plan": apakah action plan menyasar ROOT CAUSE AKHIR (why paling dalam), bukan cuma menambal gejala di why tingkat awal?
D2. "Kesesuaian PIC dengan Action Plan": apakah PIC yang ditugaskan (dan jabatannya, kalau disebutkan) masuk akal untuk jenis pekerjaan action plan itu (misal perbaikan mesin ditugaskan ke bagian teknik/maintenance, bukan ke HR/admin)?
D3. "Status FUP ke Klaim Implementasi": kalau FUP belum disetujui/approved, apakah masih masuk akal dokumen mengklaim action plan sudah terlaksana penuh dan distandardisasi? (ini red flag prosedural kalau tidak konsisten)
D4. "Rencana ke Bukti Pelaksanaan": apakah dokumentasi pelaksanaan (foto, laporan trial) menunjukkan PERSIS action plan yang direncanakan, bukan sesuatu yang berbeda?

## FASE CHECK (C1-C2)
C1. "Metodologi & SCOPE Pengukuran Awal vs Akhir": KRITIKAL: Periksa apakah cara mengukur DAN SKALA/SCOPE hasil akhir SEPADAN dengan kondisi awal! (Misal: Jika masalah awal & targetnya adalah "Mesin A", maka hasil akhirnya juga harus untuk "Mesin A". Jika hasil akhir diklaim untuk "Seluruh Pabrik" padahal target hanya 1 mesin, ini manipulasi data dan WAJIB ditandai TIDAK KONSISTEN/LEMAH).
C2. "Target Awal ke Hasil Akhir": apakah angka target awal benar-benar dijawab hasil akhir yang dilaporkan secara spesifik, tanpa pergeseran target yang tidak dijelaskan?

## FASE ACT (A1-A4)
A1. "Action Plan Efektif ke Standardisasi": apakah dokumen standar (SOP/IK/OPL/dst) yang dibuat memang RELEVAN dan mengunci action plan spesifik itu (bukan dokumen standar generik yang tidak nyambung)?
A2. "Standardisasi ke Validasi/Approval": apakah standar yang disosialisasikan (poin sosialisasi) adalah standar YANG SAMA dengan yang sudah divalidasi/disahkan, bukan draft berbeda?
A3. "Sosialisasi ke Sasaran yang Tepat": apakah pihak yang mengikuti sosialisasi (dari bukti absensi) memang pihak yang relevan/terlibat di area masalah (sesuai Who/PIC di 5W1H)? Gunakan tanggal AKTUAL pelaksanaan sosialisasi (bukan tanggal berlaku template formulir) kalau relevan untuk memeriksa urutan waktu.
A4. "Standardisasi/Action Plan ke Kelayakan Replikasi": apakah area/mesin yang diklaim direplikasi punya karakteristik yang sepadan/sejenis dengan area asal masalah (sehingga replikasi itu masuk akal secara teknis), bukan cuma diklaim "direplikasi" tanpa penjelasan kesesuaian?

Untuk tiap titik, beri verdict SALAH SATU dari: "KONSISTEN" (jelas dan masuk akal, didukung angka/isi konkret), "LEMAH" (ada tapi kurang detail/agak dipaksakan/tidak ada angka jelas, atau root cause masih bersifat "potensi"), atau "TIDAK KONSISTEN" (ada loncatan logika/manipulasi scope/tidak nyambung/tidak ditemukan).

Keluarkan HANYA JSON array valid dengan skema persis (field "fase" WAJIB salah satu dari "PLAN", "DO", "CHECK", "ACT"):
[{{"no": "P1", "fase": "PLAN", "tahap": "5G ke 5W1H", "verdict": "KONSISTEN", "temuan": "penjelasan spesifik merujuk isi dan angka konkret dari dokumen, sebutkan halaman DAN isinya"}}]"""


def prompt_kritis(laporan_ekstraksi, raw_alur):
    return (
        "Anda berperan sebagai Analis Kritis (Tinjauan Independen) yang skeptis secara metodologis dan sangat teliti dalam audit Kaizen ini. Tugas Anda mengidentifikasi kelemahan KOHERENSI dan LOGIKA, bukan cuma kelengkapan administratif, berdasarkan fakta yang ada (jangan mengarang temuan). JANGAN cuma menyebut nomor halaman — selalu jelaskan ISI KONKRET dan ANGKA yang jadi dasar analisis Anda.\n\n"
        f"Fakta Kasus:\n{laporan_ekstraksi}\n\n"
        f"HASIL AUDIT KONSISTENSI METODOLOGI PDCA:\n{raw_alur}\n\n"
        "Periksa dan pertanyakan secara spesifik, dengan mengacu ke hasil audit di atas:\n"
        "- Titik mana saja (di fase manapun) yang berstatus 'LEMAH' atau 'TIDAK KONSISTEN' — jelaskan isi temuannya dan kenapa itu masalah serius untuk kredibilitas penilaian.\n"
        "- Kelemahan bukti, celah antara masalah dan solusi, kurangnya data pendukung, manipulasi scope (target 1 mesin vs hasil 1 pabrik), atau potensi manipulasi angka saving.\n\n"
        "Sertakan alasan yang merujuk ke fakta di atas untuk tiap temuan."
    )


def prompt_konfirmatif(laporan_ekstraksi, temuan_kritis):
    return (
        "Anda berperan sebagai Analis Konfirmatif (Tinjauan Pembanding) dalam audit Kaizen ini. Tugas Anda mengevaluasi HANYA berdasarkan fakta yang tersedia, bukan asumsi baik yang tidak berdasar, untuk menyeimbangkan temuan Analisis Kritis di atas.\n\n"
        f"Fakta:\n{laporan_ekstraksi}\n\n"
        f"Temuan Analisis Kritis:\n{temuan_kritis}\n\n"
        "Bantah temuan yang tidak berdasar dan soroti nilai tambah yang sudah terbukti dari fakta di atas."
    )


def prompt_skoring(laporan_ekstraksi, raw_verifikasi, temuan_kritis, temuan_konfirmatif, raw_alur):
    return f"""Anda adalah modul Sintesis Skoring Rubrik yang wajib bersikap objektif, konsisten, dan KRITIS TERHADAP ISI — bukan cuma mengecek "ada/tidak ada elemen", tapi memverifikasi apakah isinya benar secara logika, tepat kategorinya, dan nyambung alur PDCA-nya.

ATURAN PENILAIAN:
- Beri skor SESUAI pilihan yang tersedia per kriteria (jangan beri skor di luar pilihan yang tercantum di rubrik).
- Ikuti PERSIS deskripsi tiap tingkat skor di rubrik di bawah — jangan menebak sendiri artinya.
- JANGAN PERNAH menulis justifikasi yang cuma menyebut nomor halaman tanpa penjelasan (misal "ada di halaman 24 dan 31" SAJA itu DILARANG) — selalu jelaskan ISI KONKRET dan ANGKA/MEASUREMENT yang mendasari skor itu.
- Pertimbangkan bahwa target/masalah/hasil bisa tertulis IMPLISIT (tersirat), bukan cuma yang eksplisit — cek penanda EKSPLISIT/IMPLISIT di data ekstraksi.
- Peta pemakaian bukti audit per kriteria:
  * Kriteria 1 (5G): pakai HASIL AUDIT KONSISTENSI METODOLOGI titik P1.
  * Kriteria 2 (Losses Measurement) & 5 (Target SMART): pakai titik P3.
  * Kriteria 3 (5W1H) & 4 (Visualisasi): pakai HASIL VERIFIKASI (tabel 5W1H) dan titik P2.
  * Kriteria 6 (Fishbone) & 7 (Pemetaan 4M): pakai titik P4 dan P5.
  * Kriteria 8 (Hubungan Akar Penyebab): pakai titik P6.
  * Kriteria 9 (Bukti Akar Penyebab): pakai titik P7.
  * Kriteria 10 (Ketepatan Root Cause): pakai titik P6 dan P7 bersama. JANGAN berikan skor 2 jika P7 berstatus LEMAH akibat adanya root cause yang bersifat asumtif, spekulatif, atau berupa "potensi" yang belum dibuktikan validitasnya secara teknis (misal: menebak ada reaksi kimia tanpa uji lab). Root cause final haruslah fakta teruji.
  * Kriteria 11 (Action Plan & PIC) & 12 (Rencana Perbaikan): pakai titik D1 dan D2.
  * Kriteria 13 (FUP): Berdasarkan aturan IMS Perusahaan, FUP (Form Usulan Perbaikan) ADALAH MUTLAK WAJIB untuk SEMUA jenis project improvement tanpa terkecuali, sebagai alat identifikasi risiko tersembunyi. JANGAN PERNAH memberikan skor 5 jika dokumen FUP yang sah (hasil Verifikasi Visual D: ada kop surat, judul FUP, dan sudah ditandatangani/approved) tidak dilampirkan, meskipun action plan sudah berjalan atau ada dokumen OPL/Sosialisasi. Jika tidak ada bukti FUP yang sah, SKOR WAJIB NOL (0).
  * Kriteria 14 (Pelaksanaan): pakai titik D4.
  * Kriteria 16 (Pencapaian Target): pakai titik C1 dan C2 BERSAMA — kalau metodologi pengukuran atau SCOPE/SKALA target vs hasil (C1) tidak sepadan (misal target 1 mesin diklaim hasil 1 pabrik), skor WAJIB diturunkan drastis meski angkanya kelihatan mencapai target.
  * Kriteria 17 (Pengecekan Hasil): pakai titik C1.
  * Kriteria 18 (Standardisasi) & 19 (Validasi): pakai titik A1 dan A2.
  * Kriteria 20 (Sosialisasi): pakai titik A3.
  * Kriteria 21 (Replikasi): pakai titik A4.
  Kalau titik yang relevan berstatus "TIDAK KONSISTEN" atau "LEMAH", skor kriteria itu WAJIB ikut diturunkan dan justifikasi WAJIB menyebutkan temuan spesifik dari titik tersebut (bukan cuma menyebut kode titiknya, tapi isi temuannya).
- Kalau GATE CHECK di hasil verifikasi menyatakan "TIDAK LAYAK" (bukan proyek improvement), sebutkan ini secara eksplisit di justifikasi kriteria 1-5 (tahap Plan) karena ini mempengaruhi validitas keseluruhan submission — tapi tetap beri skor per kriteria sesuai bukti yang ada (jangan otomatis nol semua tanpa dasar).
- Bersikap ketat: skor tinggi hanya untuk bukti yang benar-benar kuat, lengkap, DAN koheren secara logika.

ATURAN PENENTUAN KEBUTUHAN VALIDASI MANUAL:
Selain skor dan justifikasi, untuk SETIAP kriteria tentukan juga apakah kriteria itu PERLU DIVALIDASI MANUAL oleh asesor lapangan, dengan mengisi field "perlu_validasi_manual" ("YA" atau "TIDAK") dan "alasan_validasi_manual" (WAJIB diisi 1 kalimat spesifik kalau "YA"; kosongkan "" kalau "TIDAK"). Tandai "YA" jika salah satu berlaku:
(a) Bukti di dokumen ini bersifat implisit/tidak eksplisit sehingga interpretasinya bisa diperdebatkan.
(b) Ada perbedaan/konflik antara temuan Analisis Kritis dan Analisis Konfirmatif, atau titik Audit Konsistensi Metodologi terkait berstatus LEMAH/TIDAK KONSISTEN.
(c) Kriteria ini pada dasarnya membutuhkan verifikasi terhadap kondisi fisik/lapangan aktual yang tidak mungkin dipastikan hanya dari dokumen atau foto (misal: kesesuaian SOP yang tertempel di lokasi kerja, kepastian eksekusi PIC sesuai jadwal riil, kesesuaian teknis area tujuan replikasi).
(d) Skor bergantung pada asumsi karena data yang tersedia tidak lengkap/ambigu.
Tandai "TIDAK" jika bukti dokumen sudah eksplisit, konsisten antar agen, dan tidak memerlukan pengecekan fisik lapangan tambahan.

Referensi tipe kriteria yang secara struktural SERING (bukan otomatis SELALU) membutuhkan verifikasi lapangan — gunakan sebagai bahan pertimbangan, bukan aturan baku, karena keputusan akhir HARUS berdasarkan isi dokumen spesifik ini:
{_DAFTAR_RUJUKAN_VALIDASI_STR}

RUBRIK LENGKAP (deskripsi tiap tingkat skor):
{RUBRIK_21_POIN_DETAIL}

FAKTA HASIL EKSTRAKSI DOKUMEN:
{laporan_ekstraksi}

HASIL VERIFIKASI KELAYAKAN, 5W1H & FOTO & FUP:
{raw_verifikasi}

TEMUAN ANALISIS KRITIS (pertimbangkan temuan ini dalam penilaian):
{temuan_kritis}

TEMUAN ANALISIS KONFIRMATIF:
{temuan_konfirmatif}

Keluarkan HANYA JSON array valid, tanpa teks lain, dengan skema persis (justifikasi harus spesifik, menyebutkan ISI dan ANGKA konkret dari dokumen serta hasil audit alur logika/verifikasi, minimal 1-2 kalimat menjelaskan MENGAPA skor itu diberikan — DILARANG hanya menyebut nomor halaman tanpa penjelasan isinya; WAJIB juga isi field perlu_validasi_manual dan alasan_validasi_manual sesuai aturan di atas):
[{{"no": 1, "kriteria": "5G", "skor": 2, "justifikasi": "alasan spesifik merujuk isi dokumen dan analisis koherensi, sebutkan angka/isi konkret", "perlu_validasi_manual": "TIDAK", "alasan_validasi_manual": ""}}]

HASIL AUDIT KONSISTENSI METODOLOGI PDCA:
{raw_alur}"""


def prompt_saving(laporan_ekstraksi):
    daftar_kategori_str = ", ".join(KATEGORI_IMPACT_14)
    return f"""Anda adalah Analis Dampak Operasional yang menilai dampak operasional dari dokumen Kaizen ini secara objektif berdasarkan bukti tertulis saja.

ATURAN MEMBACA TABEL IMPACT/MANFAAT: dokumen Kaizen sering memuat tabel dengan format 'kategori impact | ambang batas skor rendah | ambang batas skor tinggi | penjelasan'. Dua kolom di tengah (misal 'Mengurangi ≤ 1%' vs 'Mengurangi >5%') adalah AMBANG BATAS/SKALA PENILAIAN GENERIK yang SELALU muncul di semua baris kategori terlepas dari relevansinya dengan proyek ini — ini BUKAN bukti pencapaian aktual. Kolom 'PENJELASAN'/'keterangan' di ujung kanan tabel adalah SATU-SATUNYA kolom yang berisi pencapaian AKTUAL proyek ini. Kalau kolom penjelasan untuk suatu kategori KOSONG SEPENUHNYA (tidak ada satu kalimat pun), kategori itu TIDAK diukur/tidak terdampak oleh proyek ini — JANGAN mengarang atau menyimpulkan pencapaian dari angka ambang batas skala pada kolom tengah.

FAKTA DOKUMEN:
{laporan_ekstraksi}

Evaluasi {len(KATEGORI_IMPACT_14)} kategori impact berikut: {daftar_kategori_str}.
Untuk setiap kategori, status HARUS salah satu dari: 'IYA' (ada dampak terbukti dengan KETERANGAN/PENJELASAN AKTUAL yang jelas di dokumen — bukan sekadar ambang batas skala penilaian), 'TIDAK' (tidak ada dampak/tidak disebutkan sama sekali, ATAU kolom penjelasan/keterangan untuk kategori itu kosong), atau 'TIDAK YAKIN' (ADA keterangan/penjelasan tapi tidak lengkap/ambigu/tidak cukup data pendukung).

SELAIN itu, tentukan juga 'Jenis Saving' berdasarkan dokumen. Pilihan statusnya adalah: 'Hard Saving' (saving finansial nyata >100 juta rupiah/tahun, terkait penurunan pemakaian gas/listrik/air/pembelian material/manpower), 'Virtual/Soft Saving' (saving tidak real, berupa opportunity loss yang dihindari, cost avoidance, material balance/stock akurasi, atau penurunan customer complaint), 'Keduanya', atau 'Tidak Ada'. 
PENTING ANTI-MANIPULASI: Anda DILARANG KERAS melabeli 'Hard Saving' jika dokumen HANYA MENCANTUMKAN ANGKA TOTAL (misal 'Saving Rp 200 Juta') tanpa ada rincian perhitungan atau parameter sebelum/sesudah yang jelas. Jika tidak ada rincian yang valid, turunkan statusnya menjadi 'Tidak Yakin' atau 'Virtual/Soft Saving'.
WAJIB JELASKAN ALASAN MENGAPA Anda mengkategorikannya sebagai Hard/Soft Saving di kolom keterangan. JANGAN KOSONGKAN keterangan untuk Jenis Saving.

Keluarkan HANYA JSON array valid dengan skema persis:
[{{"kategori": "Air", "status": "TIDAK", "keterangan": "alasan singkat merujuk dokumen"}}]"""


def prompt_feedback(laporan_ekstraksi, raw_verifikasi, raw_alur, raw_skoring):
    return f"""Anda berperan sebagai narasumber pembinaan (coaching) Kaizen yang memberikan umpan balik konstruktif untuk PESERTA kompetisi (bukan untuk juri). Bahasa harus suportif, jelas, mudah dicerna oleh peserta yang levelnya beragam (sebagian belum paham PDCA dengan baik) — kritik boleh tegas dan jujur, tapi disampaikan dengan cara yang mendidik dan tidak menjatuhkan semangat.

Untuk MASING-MASING 6 kategori tetap di bawah, isi 3 kolom: "kekuatan" (apa yang sudah bagus, sebutkan konkret — kalau memang tidak ada yang menonjol, boleh tulis "Belum ada yang menonjol di bagian ini"), "area_perbaikan" (apa yang paling perlu ditingkatkan, jelaskan KENAPA), dan "saran_konkret" (langkah nyata dan actionable yang bisa dilakukan peserta, bukan saran generik).

6 KATEGORI TETAP:
1. "Struktur & Kejelasan Penulisan" — organisasi paper, ada tidaknya typo/salah ketik yang mengganggu, konsistensi format/penomoran, kejelasan bahasa (rujuk temuan kualitas penulisan dari hasil ekstraksi bila ada; JANGAN mengangkat tanggal berlaku template dokumen kontrol sebagai contoh kesalahan penulisan).
2. "Perumusan Problem Statement" — apakah masalah dirumuskan dengan jelas, spesifik, dan didukung data (bukan cuma opini/dugaan).
3. "Kesesuaian Goal/Target dengan Objective Awal" — apakah target yang ditetapkan di awal benar-benar terjawab oleh hasil akhir, dan apakah target itu sendiri masuk akal/berdasar data. (Tegur keras jika ada manipulasi scope/skala hasil akhir).
4. "Kedalaman Analisis Akar Masalah" — kualitas fishbone dan why-why analysis: apakah benar-benar sampai ke akar masalah teknis, atau masih berupa tebakan/potensi tanpa uji coba (rujuk evaluasi kriteria 10/P7).
5. "Kekuatan Bukti & Data Pendukung" — apakah klaim-klaim (masalah, hasil, saving) didukung data/foto yang jelas (bukan foto before-after yang dimanipulasi) dan measurement yang konkret, atau banyak yang cuma klaim tanpa bukti.
6. "Standardisasi & Keberlanjutan" — apakah perbaikan ini benar-benar dikunci supaya tidak terulang (SOP/standar) dan kelengkapan dokumen FUP (rujuk jika peserta gagal melampirkan FUP resmi).

Setelah 6 kategori tetap itu, BOLEH tambahkan 0-2 baris tambahan dengan kategori "Catatan Tambahan" untuk temuan penting lain yang tidak masuk 6 kategori di atas (kalau memang ada yang signifikan; kalau tidak ada, tidak usah dipaksakan).

FAKTA EKSTRAKSI:
{laporan_ekstraksi}

VERIFIKASI VISUAL:
{raw_verifikasi}

Keluarkan HANYA JSON array valid dengan skema persis:
[{{"kategori": "Struktur & Kejelasan Penulisan", "kekuatan": "...", "area_perbaikan": "...", "saran_konkret": "..."}}]

AUDIT LOGIKA:
{raw_alur}

HASIL SKORING:
{raw_skoring}"""


# ==========================================
# 6b. KHUSUS GROQ: PROMPT RINGKAS + BATCH (agar patuh limit 8K token/menit)
# ==========================================
def muat_di_budget(kerangka, konteks, bobot, max_output):
    """kerangka memuat placeholder; konteks={placeholder: teks}. Teks dipotong proporsional bobot
    (sisa jatah bagian yang pendek dibagikan ke bagian lain) agar total prompt muat di budget token."""
    tetap = len(kerangka) - sum(len(k) for k in konteks)
    budget = int((GROQ_TPM - max_output - 150) * GROQ_CHAR_PER_TOKEN) - tetap
    budget = max(budget, 1500)
    sisa, aktif, hasil = budget, dict(konteks), {}
    while aktif:
        total_bobot = sum(bobot[k] for k in aktif)
        kecil = [k for k, v in aktif.items() if len(v) <= sisa * bobot[k] / total_bobot]
        if not kecil:
            for k, v in aktif.items():
                hasil[k] = ringkas(v, int(sisa * bobot[k] / total_bobot))
            break
        for k in kecil:
            hasil[k] = aktif.pop(k)
            sisa -= len(hasil[k])
    prompt = kerangka
    for k, v in hasil.items():
        prompt = prompt.replace(k, v)
    return prompt


def _groq_json(prompt, deskripsi, log, max_output=None, percobaan=2):
    """Panggil Groq lalu parse JSON; ulangi sekali jika JSON tidak terbaca."""
    for _ in range(percobaan):
        raw = panggil_groq(prompt, deskripsi, log, max_output=max_output)
        if not raw:
            return []  # kegagalan API sudah dilaporkan oleh panggil_groq
        items = bersihkan_dan_parse_json(raw)
        if items:
            return items
        log.write(f"⚠️ **{deskripsi} (Groq):** JSON tidak terbaca, mencoba ulang...")
    log.write(f"❌ **{deskripsi} (Groq):** JSON tetap tidak terbaca.")
    return []


# --- Pemecah hasil ekstraksi per bagian (1-9) agar tiap batch hanya membawa bagian relevan ---
_JUDUL_BAGIAN = r"(MASALAH|TARGET|FISHBONE|ANALISIS\s*5|ACTION|IMPLEMENTASI|HASIL|STANDARDISASI|KUALITAS)"


def pecah_bagian_ekstraksi(laporan):
    pola = re.compile(rf"(?im)^[#*\s_>\-]*([1-9])\s*[.)]\s*[*_\s]*{_JUDUL_BAGIAN}")
    laporan = laporan or ""
    cocok = list(pola.finditer(laporan))
    bagian = {}
    for i, m in enumerate(cocok):
        akhir = cocok[i + 1].start() if i + 1 < len(cocok) else len(laporan)
        bagian.setdefault(int(m.group(1)), laporan[m.start():akhir].strip())
    return bagian if len(bagian) >= 5 else {}  # kalau format tak terbaca → pakai seluruh teks


def ambil_bagian(laporan, bagian, nomor):
    if not bagian:
        return laporan
    teks = "\n\n".join(bagian[n] for n in nomor if n in bagian)
    return teks or laporan


# --- Audit logika: dipecah per fase ---
GRUP_ALUR = [
    {"fase": ("PLAN",), "bagian": [1, 2, 3, 4]},
    {"fase": ("DO", "CHECK", "ACT"), "bagian": [2, 5, 6, 7, 8]},
]


def _filter_alur_fase(prompt, fase_dipilih):
    awal = prompt.index("## FASE PLAN")
    akhir = prompt.index("Untuk tiap titik, beri verdict")
    blok = re.split(r"(?=## FASE )", prompt[awal:akhir])
    pilih = [b for b in blok if any(b.startswith(f"## FASE {f} ") for f in fase_dipilih)]
    catatan = "(Pada permintaan ini evaluasi HANYA titik-titik berikut; abaikan fase lain:)\n\n"
    return prompt[:awal] + catatan + "".join(pilih) + prompt[akhir:]


def groq_audit_logika(laporan, bagian, verif, log):
    kerangka_full = prompt_alur("§LAPORAN§", "§VERIF§")
    hasil = []
    for i, grup in enumerate(GRUP_ALUR, 1):
        kerangka = _filter_alur_fase(kerangka_full, grup["fase"])
        teks = ambil_bagian(laporan, bagian, grup["bagian"])
        p = muat_di_budget(kerangka, {"§LAPORAN§": teks, "§VERIF§": verif}, {"§LAPORAN§": 3, "§VERIF§": 1}, GROQ_MAX_OUTPUT)
        hasil += _groq_json(p, f"Audit Logika {i}/{len(GRUP_ALUR)}", log)
    return json.dumps(hasil, ensure_ascii=False) if hasil else ""


# --- Skoring rubrik: dipecah per kelompok kriteria, memakai prompt ringkas ---
PETA_BUKTI = {
    1: "Kriteria 1 (5G): titik audit P1.",
    2: "Kriteria 2 (Losses Measurement): titik audit P3.",
    3: "Kriteria 3 (Kelengkapan 5W1H): hasil VERIFIKASI (tabel 5W1H) dan titik audit P2.",
    4: "Kriteria 4 (Visualisasi): hasil VERIFIKASI (foto/diagram) dan titik audit P2.",
    5: "Kriteria 5 (Target SMART): titik audit P3.",
    6: "Kriteria 6 (Fishbone): titik audit P4 dan P5.",
    7: "Kriteria 7 (Pemetaan 4M): titik audit P4 dan P5.",
    8: "Kriteria 8 (Hubungan Akar Penyebab): titik audit P6.",
    9: "Kriteria 9 (Bukti Akar Penyebab): titik audit P7.",
    10: "Kriteria 10 (Ketepatan Root Cause): titik audit P6 dan P7 bersama. JANGAN beri skor 2 jika P7 LEMAH akibat root cause asumtif/spekulatif/'potensi' yang belum dibuktikan secara teknis (misal menebak reaksi kimia tanpa uji lab); root cause final harus fakta teruji.",
    11: "Kriteria 11 (Action Plan & PIC): titik audit D1 dan D2.",
    12: "Kriteria 12 (Rencana Perbaikan): titik audit D1 dan D2.",
    13: "Kriteria 13 (FUP): berdasarkan aturan IMS, FUP WAJIB untuk SEMUA project improvement. JANGAN beri skor 5 jika FUP yang sah (menurut hasil Verifikasi Visual bagian FUP: ada kop surat, judul FUP, sudah ditandatangani/approved) tidak dilampirkan, meskipun ada OPL/sosialisasi. Jika tidak ada bukti FUP yang sah, SKOR WAJIB 0.",
    14: "Kriteria 14 (Pelaksanaan Action Plan): titik audit D4.",
    15: "Kriteria 15 (Dokumentasi Pelaksanaan): bukti implementasi di ekstraksi, hasil VERIFIKASI foto, dan titik audit D4.",
    16: "Kriteria 16 (Pencapaian Target): titik audit C1 dan C2 BERSAMA — jika metodologi pengukuran atau SCOPE/SKALA target vs hasil (C1) tidak sepadan (misal target 1 mesin diklaim hasil 1 pabrik), skor WAJIB diturunkan drastis meski angka tampak mencapai target.",
    17: "Kriteria 17 (Pengecekan Hasil): titik audit C1.",
    18: "Kriteria 18 (Kelengkapan Standardisasi): titik audit A1 dan A2.",
    19: "Kriteria 19 (Validasi Standardisasi): titik audit A1 dan A2.",
    20: "Kriteria 20 (Sosialisasi): titik audit A3.",
    21: "Kriteria 21 (Replikasi): titik audit A4.",
}

GRUP_SKORING = [
    {"kriteria": [1, 2, 3, 4, 5], "bagian": [1, 2]},
    {"kriteria": [6, 7, 8, 9, 10], "bagian": [3, 4]},
    {"kriteria": [11, 12, 13, 14, 15], "bagian": [5, 6]},
    {"kriteria": [16, 17], "bagian": [2, 7]},
    {"kriteria": [18, 19, 20, 21], "bagian": [8, 5]},
]
GROQ_OUT_SKORING = 2200


def rubrik_subset(nomor):
    hasil, aktif = [], False
    for baris in RUBRIK_21_POIN_DETAIL.splitlines():
        m = re.match(r"^(\d+)\. ", baris)
        if m:
            aktif = int(m.group(1)) in nomor
        elif not baris.startswith("   "):
            aktif = False
        if aktif:
            hasil.append(baris)
    return "\n".join(hasil)


def prompt_skoring_ringkas(nomor):
    peta = "\n".join(f"- {PETA_BUKTI[n]}" for n in nomor)
    rujukan = "\n".join(
        f"  · Kriteria {n}: {KRITERIA_RUJUKAN_VALIDASI_MANUAL[n]}" for n in nomor if n in KRITERIA_RUJUKAN_VALIDASI_MANUAL
    )
    blok_rujukan = f"- Rujukan kriteria yang sering butuh verifikasi lapangan (bukan aturan baku):\n{rujukan}\n" if rujukan else ""
    daftar_nomor = ", ".join(str(n) for n in nomor)
    return f"""Anda adalah modul Sintesis Skoring Rubrik Kaizen yang objektif, ketat, dan KRITIS TERHADAP ISI — bukan sekadar mengecek "ada/tidak ada elemen", tapi memverifikasi apakah isinya benar secara logika, tepat kategorinya, dan nyambung alur PDCA-nya.

Nilai HANYA kriteria nomor: {daftar_nomor}.

ATURAN:
- Skor HARUS salah satu pilihan di rubrik; ikuti PERSIS deskripsi tiap tingkat skor.
- Justifikasi (1-2 kalimat) WAJIB menyebut ISI KONKRET dan ANGKA/MEASUREMENT dari dokumen; DILARANG hanya menyebut nomor halaman. Target/masalah/hasil bisa tertulis IMPLISIT — cek penanda EKSPLISIT/IMPLISIT di data ekstraksi.
- Bukti audit yang dipakai per kriteria:
{peta}
- Jika titik audit terkait berstatus "LEMAH" atau "TIDAK KONSISTEN", skor WAJIB diturunkan dan justifikasi WAJIB menyebut isi temuannya (bukan hanya kode titiknya).
- Jika GATE CHECK pada hasil verifikasi = "TIDAK LAYAK", sebutkan di justifikasi kriteria tahap Plan, tetapi tetap beri skor sesuai bukti (jangan otomatis nol tanpa dasar).
- Bersikap ketat: skor tinggi hanya untuk bukti yang kuat, lengkap, DAN koheren secara logika.
- Isi "perlu_validasi_manual": "YA" (plus "alasan_validasi_manual" 1 kalimat spesifik) jika: (a) bukti implisit/bisa diperdebatkan; (b) ada konflik Analisis Kritis vs Konfirmatif atau titik audit terkait LEMAH/TIDAK KONSISTEN; (c) butuh verifikasi fisik/lapangan yang tak bisa dipastikan dari dokumen/foto (SOP di lokasi, eksekusi PIC sesuai jadwal riil, kesesuaian area replikasi); (d) skor bergantung asumsi karena data tidak lengkap/ambigu. Selain itu "TIDAK" dan alasan "".
{blok_rujukan}
RUBRIK (kriteria yang dinilai):
{rubrik_subset(set(nomor))}

FAKTA EKSTRAKSI DOKUMEN (bagian relevan):
§LAPORAN§

HASIL VERIFIKASI (kelayakan, 5W1H, foto, FUP):
§VERIF§

TITIK AUDIT KONSISTENSI PDCA YANG RELEVAN:
§ALUR§

ANALISIS KRITIS:
§KRITIS§

ANALISIS KONFIRMATIF:
§KONFIRM§

Keluarkan HANYA JSON array valid (tanpa teks lain), satu objek per kriteria yang dinilai, dengan skema persis:
[{{"no": 1, "kriteria": "5G", "skor": 2, "justifikasi": "alasan spesifik dengan isi/angka konkret", "perlu_validasi_manual": "TIDAK", "alasan_validasi_manual": ""}}]"""


def alur_relevan(alur_items, nomor):
    kode = set()
    for n in nomor:
        kode |= set(re.findall(r"\b[PDCA]\d\b", PETA_BUKTI[n]))
    pilih = [it for it in alur_items if str(it.get("no", "")).strip().upper() in kode]
    return json.dumps(pilih, ensure_ascii=False) if pilih else ""


def groq_skoring(laporan, bagian, verif, kritis, konfirmatif, alur_json, log, hanya_kriteria=None):
    alur_items = bersihkan_dan_parse_json(alur_json)
    bobot = {"§LAPORAN§": 6, "§VERIF§": 2, "§ALUR§": 2, "§KRITIS§": 1, "§KONFIRM§": 1}

    def jalankan(nomor, bagian_ids, judul):
        konteks = {
            "§LAPORAN§": ambil_bagian(laporan, bagian, bagian_ids),
            "§VERIF§": verif,
            "§ALUR§": alur_relevan(alur_items, nomor) or "(tidak tersedia)",
            "§KRITIS§": kritis,
            "§KONFIRM§": konfirmatif,
        }
        p = muat_di_budget(prompt_skoring_ringkas(nomor), konteks, bobot, GROQ_OUT_SKORING)
        items = _groq_json(p, judul, log, max_output=GROQ_OUT_SKORING)
        return [it for it in items if _ke_int(it.get("no", it.get("No"))) in nomor]

    hasil = []
    for i, grup in enumerate(GRUP_SKORING, 1):
        nomor = [n for n in grup["kriteria"] if hanya_kriteria is None or n in hanya_kriteria]
        if not nomor:
            continue
        judul = f"Skoring Rubrik {i}/{len(GRUP_SKORING)} (kriteria {nomor[0]}-{nomor[-1]})"
        items = jalankan(nomor, grup["bagian"], judul)
        ada = {_ke_int(it.get("no", it.get("No"))) for it in items}
        kurang = [n for n in nomor if n not in ada]
        if kurang and not PACER.habis():
            log.write(f"🔁 **Skoring (Groq):** kriteria {kurang} belum terisi, mencoba ulang khusus kriteria itu...")
            items += jalankan(kurang, grup["bagian"], judul + " — ulang")
        hasil += items
    target = [n for g in GRUP_SKORING for n in g["kriteria"] if hanya_kriteria is None or n in hanya_kriteria]
    ada = {_ke_int(it.get("no", it.get("No"))) for it in hasil}
    hilang = [n for n in target if n not in ada]
    if hilang:
        log.write(f"⚠️ **Skoring (Groq):** kriteria belum terisi: {hilang}")
    return json.dumps(hasil, ensure_ascii=False) if hasil else ""


# --- Saving & feedback ---
def groq_saving(laporan, log):
    kerangka = prompt_saving("§LAPORAN§")
    p = muat_di_budget(kerangka, {"§LAPORAN§": laporan}, {"§LAPORAN§": 1}, GROQ_MAX_OUTPUT)
    hasil = _groq_json(p, "Analisis Saving", log)
    return json.dumps(hasil, ensure_ascii=False) if hasil else ""


def _alur_bermasalah(alur_json):
    items = bersihkan_dan_parse_json(alur_json)
    return json.dumps(
        [
            {"no": it.get("no"), "tahap": it.get("tahap"), "verdict": it.get("verdict"), "temuan": str(it.get("temuan", ""))[:300]}
            for it in items
            if str(it.get("verdict", "")).strip().upper() != "KONSISTEN"
        ],
        ensure_ascii=False,
    )


def _skor_ringkas(skor_json):
    items = bersihkan_dan_parse_json(skor_json)
    return json.dumps(
        [{"no": it.get("no"), "skor": it.get("skor"), "justifikasi": str(it.get("justifikasi", ""))[:220]} for it in items],
        ensure_ascii=False,
    )


def groq_feedback(laporan, verif, alur_json, skor_json, log):
    kerangka = prompt_feedback("§LAPORAN§", "§VERIF§", "§ALUR§", "§SKOR§")
    konteks = {
        "§LAPORAN§": laporan,
        "§VERIF§": verif,
        "§ALUR§": _alur_bermasalah(alur_json) or "(tidak tersedia)",
        "§SKOR§": _skor_ringkas(skor_json) or "(tidak tersedia)",
    }
    bobot = {"§LAPORAN§": 3, "§VERIF§": 2, "§ALUR§": 2, "§SKOR§": 2}
    p = muat_di_budget(kerangka, konteks, bobot, GROQ_MAX_OUTPUT)
    hasil = _groq_json(p, "Umpan Balik", log)
    return json.dumps(hasil, ensure_ascii=False) if hasil else ""


# ==========================================
# 7. PIPELINE MULTI-AGENT
# ==========================================
def jalankan_pipeline(uploaded_file, log):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".pdf"
    temp_path = None
    gemini_file = None
    try:
        log.write("📄 Membaca file PDF...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        log.write("☁️ Mengunggah ke Google AI Server...")
        gemini_file = client_gemini.files.upload(file=temp_path)
        mulai = time.time()
        while getattr(getattr(gemini_file, "state", None), "name", "ACTIVE") in ("PROCESSING", "PENDING"):
            if time.time() - mulai > BATAS_UPLOAD_DETIK:
                raise TimeoutError("File terlalu lama diproses di server Google (timeout). Coba unggah ulang.")
            time.sleep(3)
            gemini_file = client_gemini.files.get(name=gemini_file.name)
        if getattr(getattr(gemini_file, "state", None), "name", "") == "FAILED":
            raise RuntimeError("Google AI gagal memproses file PDF ini.")

        log.write("🔎 **[1/6] Ekstraksi bukti dokumen**")
        laporan = panggil_gemini([gemini_file, prompt_ekstraksi()], "Ekstraksi Bukti Dokumen", log, config=CONFIG_TEXT)
        if not laporan:
            raise RuntimeError("Ekstraksi dokumen gagal (respons kosong). Cek API key/model Gemini di sidebar Diagnostik.")

        log.write("🖼️ **[2/6] Verifikasi visual & FUP**")
        raw_verif = panggil_gemini([gemini_file, prompt_verifikasi()], "Verifikasi Visual & FUP", log, config=CONFIG_JSON)

        log.write(
            "🔗 **[3/6] Audit logika PDCA** (Gemini & Groq paralel). "
            "Groq diproses bertahap (batch) agar patuh limit token — bagian ini bisa makan beberapa menit."
        )
        catatan = []
        bagian = pecah_bagian_ekstraksi(laporan)
        if not bagian:
            log.write("ℹ️ Struktur bagian ekstraksi tidak terbaca; Groq memakai seluruh teks (dipotong sesuai budget).")
        p_alur = prompt_alur(laporan, raw_verif)
        raw_alur_gem, raw_alur_groq = paralel(
            log, catatan,
            lambda lg: panggil_gemini(p_alur, "Audit Logika", lg, config=CONFIG_JSON),
            lambda lg: groq_audit_logika(laporan, bagian, raw_verif, lg),
        )
        log.write(f"✅ Audit logika selesai (Gemini: {'OK' if raw_alur_gem else 'GAGAL'}, Groq: {'OK' if raw_alur_groq else 'GAGAL'})")

        log.write("🧐 **[4/6] Analisis kritis & konfirmatif**")
        kritis = panggil_gemini(prompt_kritis(laporan, raw_alur_gem), "Analisis Kritis", log)
        konfirmatif = panggil_gemini(prompt_konfirmatif(laporan, kritis), "Analisis Konfirmatif", log)

        log.write("📝 **[5/6] Skoring rubrik & analisis saving** (paralel)")
        alur_groq_pakai = raw_alur_groq
        if not alur_groq_pakai and raw_alur_gem:
            alur_groq_pakai = raw_alur_gem
            log.write("ℹ️ Audit logika Groq kosong; skoring Groq memakai audit logika Gemini sebagai bahan.")
        p_skor_gem = prompt_skoring(laporan, raw_verif, kritis, konfirmatif, raw_alur_gem)
        raw_skor_gem, raw_skor_groq = paralel(
            log, catatan,
            lambda lg: panggil_gemini(p_skor_gem, "Skoring Rubrik", lg, config=CONFIG_JSON),
            lambda lg: groq_skoring(laporan, bagian, raw_verif, kritis, konfirmatif, alur_groq_pakai, lg),
        )
        log.write(f"✅ Skoring selesai (Gemini: {'OK' if raw_skor_gem else 'GAGAL'}, Groq: {'OK' if raw_skor_groq else 'GAGAL'})")

        p_sav = prompt_saving(laporan)
        raw_sav_gem, raw_sav_groq = paralel(
            log, catatan,
            lambda lg: panggil_gemini(p_sav, "Analisis Saving", lg, config=CONFIG_JSON),
            lambda lg: groq_saving(laporan, lg),
        )
        log.write("✅ Analisis saving selesai")

        log.write("💬 **[6/6] Umpan balik peserta** (paralel)")
        p_fb_gem = prompt_feedback(laporan, raw_verif, raw_alur_gem, raw_skor_gem)
        raw_fb_gem, raw_fb_groq = paralel(
            log, catatan,
            lambda lg: panggil_gemini(p_fb_gem, "Umpan Balik", lg, config=CONFIG_JSON),
            lambda lg: groq_feedback(laporan, raw_verif, alur_groq_pakai, raw_skor_groq, lg),
        )

        return {
            "laporan": laporan, "verif": raw_verif,
            "alur_gem": raw_alur_gem, "alur_groq": raw_alur_groq,
            "kritis": kritis, "konfirmatif": konfirmatif,
            "skor_gem": raw_skor_gem, "skor_groq": raw_skor_groq,
            "sav_gem": raw_sav_gem, "sav_groq": raw_sav_groq,
            "fb_gem": raw_fb_gem, "fb_groq": raw_fb_groq,
            "catatan": catatan,
            "konteks_groq": {
                "laporan": laporan, "verif": raw_verif, "kritis": kritis,
                "konfirmatif": konfirmatif, "alur": alur_groq_pakai,
            },
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        if gemini_file:
            try:
                client_gemini.files.delete(name=gemini_file.name)
            except Exception:
                pass


def simpan_hasil(raw, nama_file):
    ss = st.session_state
    ss.nama_file = nama_file
    ss.log_error = raw.get("catatan", [])
    ss.konteks_groq = raw.get("konteks_groq", {})
    ss.df_verifikasi = buat_df(bersihkan_dan_parse_json(raw["verif"]))
    ss.df_alur_gemini = buat_df(bersihkan_dan_parse_json(raw["alur_gem"]))
    ss.df_alur_groq = buat_df(bersihkan_dan_parse_json(raw["alur_groq"]))

    ss.df_rubrik_gemini, ss.total_skor_gemini = format_tabel_rubrik(bersihkan_dan_parse_json(raw["skor_gem"]))
    ss.df_rubrik_groq, ss.total_skor_groq = format_tabel_rubrik(bersihkan_dan_parse_json(raw["skor_groq"]))
    ss.df_banding = gabungkan_rubrik(ss.df_rubrik_gemini, ss.df_rubrik_groq)

    ss.df_saving_gemini = buat_df(bersihkan_dan_parse_json(raw["sav_gem"]))
    ss.df_saving_groq = buat_df(bersihkan_dan_parse_json(raw["sav_groq"]))
    ss.df_feedback_gemini = buat_df(bersihkan_dan_parse_json(raw["fb_gem"]))
    ss.df_feedback_groq = buat_df(bersihkan_dan_parse_json(raw["fb_groq"]))

    ss.df_alur_banding = gabungkan_alur(ss.df_alur_gemini, ss.df_alur_groq)
    ss.df_saving_banding = gabungkan_saving(ss.df_saving_gemini, ss.df_saving_groq)
    ss.df_feedback_banding = gabungkan_feedback(ss.df_feedback_gemini, ss.df_feedback_groq)

    ss.transkrip = [
        {"Peran": "Ekstraksi & Visual (Gemini)", "Laporan": f"Fakta:\n{raw['laporan']}\n\nVisual:\n{raw['verif']}"},
        {"Peran": "Audit Logika (Gemini)", "Laporan": raw["alur_gem"]},
        {"Peran": "Audit Logika (Groq)", "Laporan": raw["alur_groq"]},
        {"Peran": "Tinjauan Kritis & Konfirmatif", "Laporan": f"Kritik:\n{raw['kritis']}\n\nBantahan:\n{raw['konfirmatif']}"},
        {"Peran": "Skoring (Gemini)", "Laporan": raw["skor_gem"]},
        {"Peran": "Skoring (Groq)", "Laporan": raw["skor_groq"]},
        {"Peran": "Saving (Gemini)", "Laporan": raw["sav_gem"]},
        {"Peran": "Saving (Groq)", "Laporan": raw["sav_groq"]},
        {"Peran": "Feedback (Gemini)", "Laporan": raw["fb_gem"]},
        {"Peran": "Feedback (Groq)", "Laporan": raw["fb_groq"]},
    ]


def _nomor_ada(df):
    if df is None or df.empty:
        return set()
    return set(pd.to_numeric(df["no"], errors="coerce").dropna().astype(int))


def lengkapi_skor_groq(kosong):
    """Jalankan ulang HANYA kriteria Groq yang kosong, lalu gabungkan ke tabel yang sudah ada."""
    ss = st.session_state
    k = ss.konteks_groq
    with st.status("🔁 Melengkapi skor Groq yang kosong...", expanded=True) as sb:
        try:
            bagian = pecah_bagian_ekstraksi(k["laporan"])
            raw_baru = groq_skoring(
                k["laporan"], bagian, k["verif"], k["kritis"], k["konfirmatif"], k["alur"], sb,
                hanya_kriteria=set(kosong),
            )
            items_baru = bersihkan_dan_parse_json(raw_baru)
            if not items_baru:
                sb.update(label="❌ Groq belum berhasil mengisi kriteria yang kosong (lihat pesan di atas)", state="error")
                return False
            df_baru, _ = format_tabel_rubrik(items_baru)
            df_all = pd.concat([ss.df_rubrik_groq, df_baru], ignore_index=True)
            df_all["no"] = pd.to_numeric(df_all["no"], errors="coerce").astype(int)
            df_all = df_all.drop_duplicates("no", keep="last").sort_values("no").reset_index(drop=True)
            ss.df_rubrik_groq = df_all
            ss.total_skor_groq = float(pd.to_numeric(df_all["skor"], errors="coerce").sum())
            ss.df_banding = gabungkan_rubrik(ss.df_rubrik_gemini, ss.df_rubrik_groq)
            ss.transkrip.append({"Peran": "Skoring Groq (melengkapi)", "Laporan": raw_baru})
            ss.pop("tbl_rubrik_banding", None)
            sb.update(label="✅ Selesai", state="complete")
            return True
        except Exception as e:
            sb.update(label="❌ Gagal melengkapi skor", state="error")
            st.error(f"**Pesan error:** `{e}`")
            return False


def tampilkan_banding(df, pesan_kosong="Belum ada data untuk ditampilkan."):
    """Tampilkan tabel perbandingan Gemini vs Groq + ringkasan jumlah yang sama/beda."""
    if df is None or df.empty:
        st.info(pesan_kosong)
        return
    kata_teks = ("Temuan", "Keterangan", "Kekuatan", "Area", "Saran")
    cfg = {c: st.column_config.TextColumn(width="large") for c in df.columns if any(k in c for k in kata_teks)}
    st.dataframe(df, hide_index=True, column_config=cfg, **LEBAR)
    if "Hasil Banding" in df.columns:
        h = df["Hasil Banding"]
        st.caption(
            f"✅ Sama: {int((h == '✅ Sama').sum())}  ·  ⚠️ Beda: {int((h == '⚠️ Beda').sum())}  ·  "
            f"❓ Data tidak lengkap: {int(h.astype(str).str.startswith('❓').sum())}"
        )


# ==========================================
# 8. EXPORT EXCEL
# ==========================================
def tulis_sheet(writer, df, nama):
    nama = nama[:31]
    df = bersihkan_sel(df)
    df.to_excel(writer, sheet_name=nama, index=False)
    ws = writer.sheets[nama]
    wb = writer.book
    wrap = wb.add_format({"text_wrap": True, "valign": "top"})
    dasar = {"bold": True, "font_color": "#FFFFFF", "text_wrap": True, "valign": "vcenter"}
    head = wb.add_format({**dasar, "bg_color": "#7F8C8D"})
    head_gem = wb.add_format({**dasar, "bg_color": "#5C7C99"})   # biru = Gemini
    head_groq = wb.add_format({**dasar, "bg_color": "#C98B4B"})  # oranye = Groq
    for i, col in enumerate(df.columns):
        teks = str(col)
        ws.write(0, i, teks, head_gem if "Gemini" in teks else head_groq if "Groq" in teks else head)
        panjang = max([len(teks)] + [len(str(x)) for x in df[col].head(200)])
        ws.set_column(i, i, min(max(panjang + 2, 8), 60), wrap)
        if teks == "Hasil Banding" and len(df):
            for kata, warna in (("Beda", "#FADBD8"), ("Sama", "#D5F5E3"), ("tidak lengkap", "#FCF3CF")):
                ws.conditional_format(
                    1, i, len(df), i,
                    {"type": "text", "criteria": "containing", "value": kata, "format": wb.add_format({"bg_color": warna})},
                )
    ws.freeze_panes(1, 0)


def _hitung_beda(df):
    if df is None or df.empty or "Hasil Banding" not in df.columns:
        return 0
    return int((df["Hasil Banding"] == "⚠️ Beda").sum())


def buat_excel(df_banding_final):
    ss = st.session_state
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        final_num = pd.to_numeric(df_banding_final["Skor Final (Juri)"], errors="coerce")
        ringkasan = pd.DataFrame(
            [
                ["File dokumen", ss.nama_file],
                ["Tanggal laporan", datetime.now().strftime("%Y-%m-%d %H:%M")],
                ["Model Gemini", MODEL_GEMINI],
                ["Model Groq", MODEL_GROQ],
                ["Total skor Gemini", ss.total_skor_gemini],
                ["Total skor Groq", ss.total_skor_groq],
                ["Total skor final (juri)", float(final_num.sum())],
                ["Kriteria sudah diputuskan juri", f"{int(final_num.notna().sum())} dari {len(df_banding_final)}"],
                ["Kriteria skor berbeda", int((df_banding_final["Hasil Banding"] == "⚠️ Beda").sum())],
                ["Titik audit logika berbeda verdict", _hitung_beda(ss.df_alur_banding)],
                ["Kategori saving berbeda status", _hitung_beda(ss.df_saving_banding)],
            ],
            columns=["Item", "Nilai"],
        )
        tulis_sheet(writer, ringkasan, "Ringkasan")
        urutan = [
            (ss.df_verifikasi, "1. Verifikasi Visual"),
            (ss.df_alur_banding, "2. Alur Logika Perbandingan"),
            (df_banding_final, "3. Rubrik Perbandingan"),
            (ss.df_saving_banding, "4. Saving Perbandingan"),
            (ss.df_feedback_banding, "5. Feedback Perbandingan"),
        ]
        for df, nama in urutan:
            if df is not None and not df.empty:
                tulis_sheet(writer, df, nama)
    return output.getvalue()


# ==========================================
# 9. ALUR UNGGAH & EKSEKUSI
# ==========================================
uploaded_file = st.file_uploader("Pilih file PDF Kaizen", type="pdf")

if uploaded_file is not None and not st.session_state.proses_selesai:
    st.caption(
        "⏱️ Estimasi 10–15 menit per dokumen. "
        "Progres bisa dipantau di panel di bawah. Apabila ada kegagalan sistem, mohon berikan jeda waktu untuk "request er menit" pada sistem API."
    )
    if st.button("🚀 Mulai Penilaian"):
        berhasil = False
        with st.status("🤖 Sistem sedang memproses...", expanded=True) as status_box:
            try:
                raw = jalankan_pipeline(uploaded_file, status_box)
                simpan_hasil(raw, uploaded_file.name)
                st.session_state.proses_selesai = True
                status_box.update(label="✅ Analisis Dual-AI selesai!", state="complete")
                berhasil = True
            except Exception as e:
                status_box.update(label="❌ Terjadi Kesalahan", state="error")
                st.error(f"**Pesan error:** `{e}`")
                with st.expander("🔍 Detail teknis (traceback lengkap)"):
                    st.code(traceback.format_exc())
        if berhasil:
            st.rerun()

# ==========================================
# 10. HASIL PENILAIAN
# ==========================================
if st.session_state.proses_selesai:
    ss = st.session_state
    st.success("Analisis Dual-AI selesai! Silakan bandingkan penalaran Gemini dan Groq di bawah.")

    kosong_gem = [n for n in RUBRIK_META if n not in _nomor_ada(ss.df_rubrik_gemini)]
    kosong_groq = [n for n in RUBRIK_META if n not in _nomor_ada(ss.df_rubrik_groq)]
    if kosong_gem or kosong_groq:
        bagian_pesan = []
        if kosong_gem:
            bagian_pesan.append(f"Gemini: kriteria {kosong_gem}")
        if kosong_groq:
            bagian_pesan.append(f"Groq: kriteria {kosong_groq}")
        st.warning("Skor rubrik belum lengkap — " + "; ".join(bagian_pesan) + ".")
    if ss.log_error:
        with st.expander("🩺 Catatan & penyebab error dari Gemini/Groq", expanded=bool(kosong_gem or kosong_groq)):
            for pesan in ss.log_error:
                st.markdown(f"- {pesan}")
    elif kosong_gem or kosong_groq:
        st.info(
            "Tidak ada error API yang tercatat — artinya model membalas tetapi JSON-nya tidak terbaca. "
            "Cek 'Transkrip Lengkap' untuk melihat isi balasan mentahnya."
        )
    if kosong_groq and ss.konteks_groq:
        st.caption(
            "Tombol ini hanya menjalankan ulang kriteria Groq yang kosong (tanpa mengulang seluruh dokumen). "
            "Isian 'Skor Final (Juri)' yang sudah Anda ketik akan ter-reset, jadi pakai sebelum mulai memutuskan."
        )
        if st.button("🔁 Lengkapi skor Groq yang kosong"):
            if lengkapi_skor_groq(kosong_groq):
                st.rerun()

    st.subheader("🔍 1. Fakta Observasi: Verifikasi Kelayakan, 5W1H & FUP")
    st.caption("Fakta dasar yang diekstrak oleh Gemini (sebagai Mata) dan dipakai bersama oleh kedua AI.")
    st.data_editor(ss.df_verifikasi, num_rows="dynamic", key="tbl_verifikasi", **LEBAR)

    st.subheader("🔗 2. Audit Konsistensi Metodologi PDCA (Golden Thread)")
    st.caption("Verdict Gemini dan Groq untuk tiap titik sambungan PDCA, berdampingan dalam satu tabel.")
    tampilkan_banding(ss.df_alur_banding)

    st.subheader("📝 3. Tabel Validasi Rubrik (Keputusan Akhir)")
    st.caption(
        "Tab pertama menampilkan skor Gemini dan Groq berdampingan dalam satu tabel. "
        "'Skor Final (Juri)' otomatis terisi bila kedua AI sepakat; bila berbeda, kolom dibiarkan kosong untuk Anda putuskan."
    )
    tab_banding, tab_gem, tab_groq = st.tabs(["📊 Perbandingan (1 Tabel)", "🤖 Detail GEMINI", "🚀 Detail GROQ"])

    with tab_banding:
        df_b = ss.df_banding
        kolom_terkunci = [c for c in df_b.columns if c not in ("Skor Final (Juri)", "Catatan Validator")]
        edited_banding = st.data_editor(
            df_b,
            key="tbl_rubrik_banding",
            disabled=kolom_terkunci,
            hide_index=True,
            column_config={
                "No": st.column_config.NumberColumn(width="small", format="%d"),
                "Tahap": st.column_config.TextColumn(width="small"),
                "Kriteria": st.column_config.TextColumn(width="medium"),
                "Skor Gemini": st.column_config.NumberColumn(format="%g", width="small"),
                "Skor Groq": st.column_config.NumberColumn(format="%g", width="small"),
                "Hasil Banding": st.column_config.TextColumn(width="small"),
                "Status Validasi": st.column_config.TextColumn(width="medium"),
                "Alasan Validasi Manual": st.column_config.TextColumn(width="large"),
                "Skor Final (Juri)": st.column_config.NumberColumn(
                    min_value=0, max_value=8, step=1, format="%g", width="small",
                    help="Isi/ubah skor akhir sesuai keputusan juri (mengikuti pilihan skor rubrik).",
                ),
                "Catatan Validator": st.column_config.TextColumn(width="medium"),
                "Justifikasi Gemini": st.column_config.TextColumn(width="large"),
                "Justifikasi Groq": st.column_config.TextColumn(width="large"),
            },
            **LEBAR,
        )
        final_num = pd.to_numeric(edited_banding["Skor Final (Juri)"], errors="coerce")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Skor Gemini", f"{ss.total_skor_gemini:.0f}")
        m2.metric("Total Skor Groq", f"{ss.total_skor_groq:.0f}")
        m3.metric("Kriteria Skor Berbeda", int((edited_banding["Hasil Banding"] == "⚠️ Beda").sum()))
        m4.metric(
            "Total Skor Final (Juri)",
            f"{final_num.sum():.0f}",
            help=f"Terisi {int(final_num.notna().sum())} dari {len(edited_banding)} kriteria",
        )
        # Peringatan jika skor juri di luar pilihan rubrik
        salah = [
            int(r["No"]) for _, r in edited_banding.iterrows()
            if pd.notna(r["Skor Final (Juri)"]) and int(r["Skor Final (Juri)"]) not in RUBRIK_META[int(r["No"])][2]
        ]
        if salah:
            st.warning(f"Skor final di luar pilihan rubrik pada kriteria nomor: {salah}")

    with tab_gem:
        st.metric("Total Skor Rubrik (Gemini)", f"{ss.total_skor_gemini:.0f}")
        st.dataframe(ss.df_rubrik_gemini.drop(columns=["perlu_manual"], errors="ignore"), **LEBAR)
    with tab_groq:
        st.metric("Total Skor Rubrik (Groq)", f"{ss.total_skor_groq:.0f}")
        st.dataframe(ss.df_rubrik_groq.drop(columns=["perlu_manual"], errors="ignore"), **LEBAR)

    st.subheader("💰 4. Tabel Validasi Impact & Saving (14 Kategori)")
    st.caption("Status dan keterangan dampak per kategori dari Gemini dan Groq, berdampingan dalam satu tabel.")
    tampilkan_banding(ss.df_saving_banding)

    st.subheader("💬 5. Feedback & Saran untuk Peserta")
    st.caption("Umpan balik untuk peserta dari kedua AI per kategori, berdampingan dalam satu tabel.")
    tampilkan_banding(ss.df_feedback_banding)

    with st.expander("📜 Lihat Transkrip Lengkap"):
        for entri in ss.transkrip:
            st.markdown(f"**{entri['Peran']}**")
            st.text(entri["Laporan"] or "(kosong)")
            st.divider()

    nama_aman = re.sub(r"[^\w\-.]+", "_", os.path.splitext(ss.nama_file)[0])
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Unduh Laporan Perbandingan Lengkap (Excel)",
            data=buat_excel(edited_banding),
            file_name=f"Laporan_Perbandingan_{nama_aman}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col2:
        if st.button("🔄 Unggah Dokumen Baru (Reset)"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
