import io
import json
import os
import re
import tempfile
import time
import traceback

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from groq import Groq

# ==========================================
# 1. KONFIGURASI HALAMAN & TEMA (UI/UX)
# ==========================================
st.set_page_config(
    page_title="Portal Validasi Kaizen", page_icon="🏢", layout="wide"
)

st.markdown(
    """
    <style>
    /* Sembunyikan menu bawaan agar lebih rapi */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Spacing container utama */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Judul H1 dengan warna Slate Blue yang profesional namun lembut */
    h1 { color: #5C7C99; text-align: center; font-family: 'Nunito', 'Segoe UI', sans-serif; font-weight: 700; margin-bottom: 0.5rem;}
    
    /* Gaya Tombol Modern & Pastel (Ada efek melayang saat di-hover) */
    .stButton>button { 
        background-color: #A3B9D2; 
        color: white; 
        border-radius: 8px; 
        border: none; 
        padding: 0.6rem 1.2rem; 
        font-weight: 600; 
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stButton>button:hover { 
        background-color: #8BA3C7; 
        color: white; 
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    /* Badge manual validasi (Warna pastel pink-merah yang tidak mencolok) */
    .manual-badge { 
        background-color: #FDF1F0; 
        color: #D46B6B; 
        padding: 4px 10px; 
        border-radius: 12px; 
        font-size: 0.8rem; 
        font-weight: 600; 
        border: 1px solid #F9DEDC;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🏢 Portal Validasi Kaizen (Dual-AI Judge)")
st.markdown(
    "<p style='text-align: center; color: #7F8C8D; font-size: 1.1rem; font-weight: 400; margin-bottom: 2rem;'>Unggah"
    " dokumen evaluasi, bandingkan analisis Gemini vs Llama-3 (Groq) secara <i>apple-to-apple</i>, lalu lakukan validasi akhir secara manual.</p>",
    unsafe_allow_html=True,
)
st.divider()

# ==========================================
# INISIALISASI API KEY
# ==========================================
try:
    API_KEY_GEMINI = st.secrets["GEMINI_API_KEY"].strip()
    client_gemini = genai.Client(api_key=API_KEY_GEMINI)
    
    API_KEY_GROQ = st.secrets["GROQ_API_KEY"].strip()
    client_groq = Groq(api_key=API_KEY_GROQ)
except Exception as e:
    st.error(f"Gagal memuat API Key. Pastikan GEMINI_API_KEY dan GROQ_API_KEY tersedia di Secrets. Detail: {e}")
    st.stop()

MODEL_GEMINI = "gemini-3.5-flash-lite"
MODEL_GROQ = "llama3-70b-8192e"

GENERATION_CONFIG_TEXT = types.GenerateContentConfig(
    seed=42, thinking_config=types.ThinkingConfig(thinking_level="medium")
)
GENERATION_CONFIG_JSON = types.GenerateContentConfig(
    seed=42, thinking_config=types.ThinkingConfig(thinking_level="medium"), response_mime_type="application/json"
)

KRITERIA_RUJUKAN_VALIDASI_MANUAL = {
    7: "Pemetaan 4M — verifikasi kesesuaian dengan kondisi mesin/area aktual di lapangan",
    10: "Ketepatan Root Cause — memerlukan justifikasi teknis dari asesor lapangan",
    11: "Action Plan & PIC — konfirmasi PIC dan jadwal riil pelaksanaan",
    18: "Kelengkapan Standardisasi — pengecekan dokumen fisik/SOP terbaru di lokasi kerja",
    19: "Validasi Standardisasi — verifikasi implementasi standar di lapangan",
    21: "Replikasi — konfirmasi area lain yang benar-benar direplikasi",
}
_DAFTAR_RUJUKAN_VALIDASI_STR = "\n".join(f"- Kriteria {no}: {alasan}" for no, alasan in KRITERIA_RUJUKAN_VALIDASI_MANUAL.items())

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
# 2. INISIALISASI MEMORI SESI (SESSION STATE)
# ==========================================
if "proses_selesai" not in st.session_state: st.session_state.proses_selesai = False
if "df_verifikasi" not in st.session_state: st.session_state.df_verifikasi = pd.DataFrame()

if "df_alur_gemini" not in st.session_state: st.session_state.df_alur_gemini = pd.DataFrame()
if "df_rubrik_gemini" not in st.session_state: st.session_state.df_rubrik_gemini = pd.DataFrame()
if "df_saving_gemini" not in st.session_state: st.session_state.df_saving_gemini = pd.DataFrame()
if "df_feedback_gemini" not in st.session_state: st.session_state.df_feedback_gemini = pd.DataFrame()
if "total_skor_gemini" not in st.session_state: st.session_state.total_skor_gemini = 0.0

if "df_alur_groq" not in st.session_state: st.session_state.df_alur_groq = pd.DataFrame()
if "df_rubrik_groq" not in st.session_state: st.session_state.df_rubrik_groq = pd.DataFrame()
if "df_saving_groq" not in st.session_state: st.session_state.df_saving_groq = pd.DataFrame()
if "df_feedback_groq" not in st.session_state: st.session_state.df_feedback_groq = pd.DataFrame()
if "total_skor_groq" not in st.session_state: st.session_state.total_skor_groq = 0.0

if "transkrip" not in st.session_state: st.session_state.transkrip = []
if "nama_file" not in st.session_state: st.session_state.nama_file = "Dokumen_Kaizen"

# ==========================================
# 3. FUNGSI MESIN AI & PARSER (ANTI-ERROR)
# ==========================================
def panggil_ai_dengan_retry(contents, deskripsi_agen, log_ui, config=None, maksimal_percobaan=3):
    config = config or GENERATION_CONFIG_TEXT
    for percobaan in range(maksimal_percobaan):
        try:
            log_ui.write(f"⏳ **{deskripsi_agen} (Gemini):** Sedang menganalisis...")
            response = client_gemini.models.generate_content(model=MODEL_GEMINI, contents=contents, config=config)
            teks_hasil = response.text if response and response.text else ""
            if not teks_hasil:
                time.sleep(8)
                continue
            log_ui.write(f"✅ **{deskripsi_agen} (Gemini):** Selesai! Pendinginan 15 detik...")
            time.sleep(15)
            return teks_hasil
        except Exception as e:
            if "429" in str(e).upper() or "RESOURCE_EXHAUSTED" in str(e).upper():
                log_ui.write(f"⚠️ **Gemini:** Kena limit rate. Menunggu 65 detik...")
                time.sleep(65)
            else:
                log_ui.write(f"⚠️ **Gemini:** Error: {e}. Mencoba ulang...")
                time.sleep(10)
    return ""

def panggil_groq_dengan_retry(prompt_text, deskripsi_agen, log_ui, maksimal_percobaan=3):
    for percobaan in range(maksimal_percobaan):
        try:
            log_ui.write(f"⏳ **{deskripsi_agen} (Groq Llama 3):** Sedang mengevaluasi super cepat...")
            response = client_groq.chat.completions.create(
                model=MODEL_GROQ,
                messages=[
                    {"role": "system", "content": "Anda adalah asisten auditor Kaizen tingkat senior. Wajib merespons HANYA dengan format JSON Array valid (dimulai dengan [ dan diakhiri dengan ]), tanpa teks markdown tambahan."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.2
            )
            teks_hasil = response.choices[0].message.content
            log_ui.write(f"✅ **{deskripsi_agen} (Groq Llama 3):** Selesai dalam sekejap!")
            time.sleep(2)
            return teks_hasil
        except Exception as e:
            if "429" in str(e):
                log_ui.write(f"⚠️ **Groq:** Limit tercapai. Menunggu 10 detik...")
                time.sleep(10)
            else:
                log_ui.write(f"⚠️ **Groq:** Error: {e}. Mencoba ulang...")
                time.sleep(5)
    return ""

def bersihkan_dan_parse_json(teks_raw):
    if not teks_raw: return []
    teks_bersih = re.sub(r"```json", "", teks_raw, flags=re.IGNORECASE)
    teks_bersih = re.sub(r"```", "", teks_bersih).strip()
    
    start_idx = teks_bersih.find('[')
    end_idx = teks_bersih.rfind(']')
    
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        teks_array = teks_bersih[start_idx:end_idx+1]
        try:
            return json.loads(teks_array)
        except json.JSONDecodeError:
            pass

    try:
        data = json.loads(teks_bersih)
        if isinstance(data, dict):
            for _key, value in data.items():
                if isinstance(value, list): return value
            return [data]
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        pass
        
    return []

def tentukan_validasi_manual(item, nomor_kriteria):
    nilai_mentah = item.get("perlu_validasi_manual")
    if nilai_mentah is None:
        perlu = nomor_kriteria in KRITERIA_RUJUKAN_VALIDASI_MANUAL
        alasan = KRITERIA_RUJUKAN_VALIDASI_MANUAL.get(nomor_kriteria, "") if perlu else ""
        return perlu, alasan

    perlu = str(nilai_mentah).strip().upper() in ("YA", "TRUE", "1", "YES")
    alasan = str(item.get("alasan_validasi_manual", "")).strip() if perlu else ""
    return perlu, alasan

def format_tabel_rubrik(json_data):
    total_skor = 0.0
    for item in json_data:
        nomor_kriteria = item.get("no", item.get("No", item.get("nomor", 0)))
        try: nomor_kriteria = int(nomor_kriteria)
        except: nomor_kriteria = 0
        skor_item = item.get("skor", item.get("score", None))
        try: total_skor += float(skor_item)
        except: pass
        
        perlu_validasi, alasan_validasi = tentukan_validasi_manual(item, nomor_kriteria)
        item["status validasi"] = "⚠️ VALIDASI MANUAL" if perlu_validasi else "OTOMATIS AI"
        item["alasan_validasi_manual"] = alasan_validasi if perlu_validasi else ""
        item["skor"] = skor_item
        item["catatan_validator"] = ""
    
    df = pd.DataFrame(json_data)
    if not df.empty:
        df.columns = df.columns.str.lower().str.strip()
        df.rename(columns={"nomor": "no", "score": "skor", "nilai": "skor", "alasan": "justifikasi", "keterangan": "justifikasi"}, inplace=True)
        kolom_urutan = ["no", "kriteria", "status validasi", "alasan_validasi_manual", "skor", "justifikasi", "catatan_validator"]
        cols = [c for c in kolom_urutan if c in df.columns]
        sisa = [c for c in df.columns if c not in cols]
        df = df[cols + sisa]
    return df, total_skor

# ==========================================
# 4. ALUR UNGGAH & EKSEKUSI MULTI-AGENT
# ==========================================
uploaded_file = st.file_uploader("Pilih file PDF Kaizen", type="pdf")

if uploaded_file is not None and not st.session_state.proses_selesai:
    if st.button("🚀 Mulai Penilaian AI (Gemini + Groq)"):
        st.session_state.nama_file = uploaded_file.name

        with st.status("🤖 AI Multi-Agent sedang bekerja...", expanded=True) as status_box:
            suffix = os.path.splitext(uploaded_file.name)[1] or ".pdf"
            temp_path = None
            gemini_file = None
            try:
                status_box.write("📄 Membaca berkas PDF...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_path = tmp.name

                status_box.write("☁️️ Mengunggah berkas ke Google AI Server...")
                gemini_file = client_gemini.files.upload(file=temp_path)
                while gemini_file.state.name in ["PROCESSING", "PENDING"]:
                    time.sleep(4)
                    gemini_file = client_gemini.files.get(name=gemini_file.name)

                # --- 1. Ekstraksi (HANYA GEMINI - Sebagai Mata) ---
                prompt_1 = (
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
                laporan_ekstraksi = panggil_ai_dengan_retry([gemini_file, prompt_1], "Ekstraksi Bukti Dokumen", status_box)

                # --- 2. Verifikasi Visual (HANYA GEMINI - Sebagai Mata) ---
                prompt_verifikasi = (
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
                raw_verifikasi = panggil_ai_dengan_retry([gemini_file, prompt_verifikasi], "Verifikasi Visual & FUP", status_box, config=GENERATION_CONFIG_JSON)

                # --- 3. Audit Logika PDCA (GEMINI & GROQ) ---
                prompt_alur = f"""Anda adalah Analis Audit Konsistensi Metodologi PDCA (QC-Story) yang menelusuri "benang merah" (golden thread): apakah tiap tools di tiap fase PDCA benar-benar tersambung MASUK AKAL secara teknis/operasional ke tools sebelum dan sesudahnya — bukan cuma sama-sama ada di dokumen.

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
                
                raw_alur_gemini = panggil_ai_dengan_retry(prompt_alur, "Audit Logika", status_box, config=GENERATION_CONFIG_JSON)
                raw_alur_groq = panggil_groq_dengan_retry(prompt_alur, "Audit Logika", status_box)

                # --- 4 & 5. Analisis Kritis & Konfirmatif (HANYA GEMINI - Sebagai bahan skoring) ---
                prompt_2 = (
                    "Anda berperan sebagai Analis Kritis (Tinjauan Independen) yang skeptis secara metodologis dan sangat teliti dalam audit Kaizen ini. Tugas Anda mengidentifikasi kelemahan KOHERENSI dan LOGIKA, bukan cuma kelengkapan administratif, berdasarkan fakta yang ada (jangan mengarang temuan). JANGAN cuma menyebut nomor halaman — selalu jelaskan ISI KONKRET dan ANGKA yang jadi dasar analisis Anda.\n\n"
                    f"Fakta Kasus:\n{laporan_ekstraksi}\n\n"
                    f"HASIL AUDIT KONSISTENSI METODOLOGI PDCA:\n{raw_alur_gemini}\n\n"
                    "Periksa dan pertanyakan secara spesifik, dengan mengacu ke hasil audit di atas:\n"
                    "- Titik mana saja (di fase manapun) yang berstatus 'LEMAH' atau 'TIDAK KONSISTEN' — jelaskan isi temuannya dan kenapa itu masalah serius untuk kredibilitas penilaian.\n"
                    "- Kelemahan bukti, celah antara masalah dan solusi, kurangnya data pendukung, manipulasi scope (target 1 mesin vs hasil 1 pabrik), atau potensi manipulasi angka saving.\n\n"
                    "Sertakan alasan yang merujuk ke fakta di atas untuk tiap temuan."
                )
                temuan_analisis_kritis = panggil_ai_dengan_retry(prompt_2, "Analisis Kritis", status_box)

                prompt_3 = (
                    "Anda berperan sebagai Analis Konfirmatif (Tinjauan Pembanding) dalam audit Kaizen ini. Tugas Anda mengevaluasi HANYA berdasarkan fakta yang tersedia, bukan asumsi baik yang tidak berdasar, untuk menyeimbangkan temuan Analisis Kritis di atas.\n\n"
                    f"Fakta:\n{laporan_ekstraksi}\n\n"
                    f"Temuan Analisis Kritis:\n{temuan_analisis_kritis}\n\n"
                    "Bantah temuan yang tidak berdasar dan soroti nilai tambah yang sudah terbukti dari fakta di atas."
                )
                temuan_analisis_konfirmatif = panggil_ai_dengan_retry(prompt_3, "Analisis Konfirmatif", status_box)

                # --- 6. Sintesis Skoring Rubrik (GEMINI & GROQ) ---
                prompt_skoring_base = f"""Anda adalah modul Sintesis Skoring Rubrik yang wajib bersikap objektif, konsisten, dan KRITIS TERHADAP ISI — bukan cuma mengecek "ada/tidak ada elemen", tapi memverifikasi apakah isinya benar secara logika, tepat kategorinya, dan nyambung alur PDCA-nya.

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
{temuan_analisis_kritis}

TEMUAN ANALISIS KONFIRMATIF:
{temuan_analisis_konfirmatif}

Keluarkan HANYA JSON array valid, tanpa teks lain, dengan skema persis (justifikasi harus spesifik, menyebutkan ISI dan ANGKA konkret dari dokumen serta hasil audit alur logika/verifikasi, minimal 1-2 kalimat menjelaskan MENGAPA skor itu diberikan — DILARANG hanya menyebut nomor halaman tanpa penjelasan isinya; WAJIB juga isi field perlu_validasi_manual dan alasan_validasi_manual sesuai aturan di atas):
[{{"no": 1, "kriteria": "5G", "skor": 2, "justifikasi": "alasan spesifik merujuk isi dokumen dan analisis koherensi, sebutkan angka/isi konkret", "perlu_validasi_manual": "TIDAK", "alasan_validasi_manual": ""}}]"""
                
                raw_skoring_gemini = panggil_ai_dengan_retry(f"{prompt_skoring_base}\n\nHASIL AUDIT KONSISTENSI METODOLOGI PDCA:\n{raw_alur_gemini}", "Skoring Rubrik", status_box, config=GENERATION_CONFIG_JSON)
                raw_skoring_groq = panggil_groq_dengan_retry(f"{prompt_skoring_base}\n\nHASIL AUDIT KONSISTENSI METODOLOGI PDCA:\n{raw_alur_groq}", "Skoring Rubrik", status_box)

                # --- 7. Analisis Dampak & Saving (GEMINI & GROQ) ---
                daftar_kategori_str = ", ".join(KATEGORI_IMPACT_14)
                prompt_saving = f"""Anda adalah Analis Dampak Operasional yang menilai dampak operasional dari dokumen Kaizen ini secara objektif berdasarkan bukti tertulis saja.

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
                
                raw_saving_gemini = panggil_ai_dengan_retry(prompt_saving, "Analisis Saving", status_box, config=GENERATION_CONFIG_JSON)
                raw_saving_groq = panggil_groq_dengan_retry(prompt_saving, "Analisis Saving", status_box)

                # --- 8. Feedback Peserta (GEMINI & GROQ) ---
                prompt_feedback_base = f"""Anda berperan sebagai narasumber pembinaan (coaching) Kaizen yang memberikan umpan balik konstruktif untuk PESERTA kompetisi (bukan untuk juri). Bahasa harus suportif, jelas, mudah dicerna oleh peserta yang levelnya beragam (sebagian belum paham PDCA dengan baik) — kritik boleh tegas dan jujur, tapi disampaikan dengan cara yang mendidik dan tidak menjatuhkan semangat.

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
[{{"kategori": "Struktur & Kejelasan Penulisan", "kekuatan": "...", "area_perbaikan": "...", "saran_konkret": "..."}}]"""
                
                raw_feedback_gemini = panggil_ai_dengan_retry(f"{prompt_feedback_base}\n\nAUDIT LOGIKA:\n{raw_alur_gemini}\n\nHASIL SKORING:\n{raw_skoring_gemini}", "Umpan Balik", status_box, config=GENERATION_CONFIG_JSON)
                raw_feedback_groq = panggil_groq_dengan_retry(f"{prompt_feedback_base}\n\nAUDIT LOGIKA:\n{raw_alur_groq}\n\nHASIL SKORING:\n{raw_skoring_groq}", "Umpan Balik", status_box)

                status_box.update(label="✅ Analisis Agen Ganda Selesai!", state="complete")

                # --- PEMROSESAN TABEL ---
                st.session_state.df_verifikasi = pd.DataFrame(bersihkan_dan_parse_json(raw_verifikasi))
                
                st.session_state.df_alur_gemini = pd.DataFrame(bersihkan_dan_parse_json(raw_alur_gemini))
                st.session_state.df_alur_groq = pd.DataFrame(bersihkan_dan_parse_json(raw_alur_groq))
                
                df_gem, skor_gem = format_tabel_rubrik(bersihkan_dan_parse_json(raw_skoring_gemini))
                df_groq, skor_groq = format_tabel_rubrik(bersihkan_dan_parse_json(raw_skoring_groq))
                st.session_state.df_rubrik_gemini = df_gem
                st.session_state.total_skor_gemini = skor_gem
                st.session_state.df_rubrik_groq = df_groq
                st.session_state.total_skor_groq = skor_groq
                
                st.session_state.df_saving_gemini = pd.DataFrame(bersihkan_dan_parse_json(raw_saving_gemini))
                st.session_state.df_saving_groq = pd.DataFrame(bersihkan_dan_parse_json(raw_saving_groq))
                
                st.session_state.df_feedback_gemini = pd.DataFrame(bersihkan_dan_parse_json(raw_feedback_gemini))
                st.session_state.df_feedback_groq = pd.DataFrame(bersihkan_dan_parse_json(raw_feedback_groq))

                st.session_state.transkrip = [
                    {"Peran": "Ekstraksi & Visual (Gemini Mata)", "Laporan": f"Fakta:\n{laporan_ekstraksi}\n\nVisual:\n{raw_verifikasi}"},
                    {"Peran": "Audit Logika (Gemini)", "Laporan": raw_alur_gemini},
                    {"Peran": "Audit Logika (Groq Llama)", "Laporan": raw_alur_groq},
                    {"Peran": "Tinjauan Kritis & Konfirmatif", "Laporan": f"Kritik:\n{temuan_analisis_kritis}\n\nBantahan:\n{temuan_analisis_konfirmatif}"},
                    {"Peran": "Skoring (Gemini)", "Laporan": raw_skoring_gemini},
                    {"Peran": "Skoring (Groq Llama)", "Laporan": raw_skoring_groq},
                    {"Peran": "Saving (Gemini)", "Laporan": raw_saving_gemini},
                    {"Peran": "Saving (Groq Llama)", "Laporan": raw_saving_groq},
                    {"Peran": "Feedback (Gemini)", "Laporan": raw_feedback_gemini},
                    {"Peran": "Feedback (Groq Llama)", "Laporan": raw_feedback_groq},
                ]
                st.session_state.proses_selesai = True
                st.rerun()

            except Exception as e:
                status_box.update(label="❌ Terjadi Kesalahan", state="error")
                st.error(f"**Pesan Error:** `{e}`")
                with st.expander("🔍 Detail teknis (traceback lengkap)"): st.code(traceback.format_exc())
            finally:
                if temp_path and os.path.exists(temp_path):
                    try: os.remove(temp_path)
                    except OSError: pass
                if gemini_file:
                    try: client_gemini.files.delete(name=gemini_file.name)
                    except Exception: pass

# ==========================================
# 5. HASIL PENILAIAN & UI TAB BERSANDING
# ==========================================
if st.session_state.proses_selesai:
    st.success("Analisis Dual-AI selesai! Silakan bandingkan penalaran Gemini dan Groq Llama 3 di bawah ini.")

    st.subheader("🔍 1. Fakta Observasi: Verifikasi Kelayakan, 5W1H & FUP")
    st.caption("Fakta dasar yang diekstrak oleh Gemini (sebagai Mata) dan disetujui bersama oleh kedua AI.")
    st.data_editor(st.session_state.df_verifikasi, num_rows="dynamic", use_container_width=True)

    st.subheader("🔗 2. Audit Konsistensi Metodologi PDCA (Golden Thread)")
    tab_alur_gemini, tab_alur_groq = st.tabs(["🤖 Evaluasi GEMINI", "🦙 Evaluasi GROQ LLAMA 3"])
    with tab_alur_gemini: st.data_editor(st.session_state.df_alur_gemini, num_rows="dynamic", use_container_width=True, key="tbl_alur_gemini")
    with tab_alur_groq: st.data_editor(st.session_state.df_alur_groq, num_rows="dynamic", use_container_width=True, key="tbl_alur_groq")

    st.subheader("📝 3. Tabel Validasi Rubrik (Keputusan Akhir)")
    st.caption("Manajer/Juri bertindak sebagai Hakim. Silakan sesuaikan kolom **skor** setelah mempertimbangkan debat argumen dari kedua AI.")
    tab_rub_gemini, tab_rub_groq = st.tabs(["🤖 Skoring GEMINI", "🦙 Skoring GROQ LLAMA 3"])
    with tab_rub_gemini:
        st.metric("Total Skor Rubrik (Gemini)", f"{st.session_state.total_skor_gemini:.0f}")
        edited_rubrik_gemini = st.data_editor(st.session_state.df_rubrik_gemini, num_rows="dynamic", use_container_width=True, key="tbl_rubrik_gemini")
    with tab_rub_groq:
        st.metric("Total Skor Rubrik (Groq Llama)", f"{st.session_state.total_skor_groq:.0f}")
        edited_rubrik_groq = st.data_editor(st.session_state.df_rubrik_groq, num_rows="dynamic", use_container_width=True, key="tbl_rubrik_groq")

    st.subheader("💰 4. Tabel Validasi Impact & Saving (14 Kategori)")
    tab_sav_gemini, tab_sav_groq = st.tabs(["🤖 Analisis Saving GEMINI", "🦙 Analisis Saving GROQ LLAMA 3"])
    with tab_sav_gemini: st.data_editor(st.session_state.df_saving_gemini, num_rows="dynamic", use_container_width=True, key="tbl_saving_gemini")
    with tab_sav_groq: st.data_editor(st.session_state.df_saving_groq, num_rows="dynamic", use_container_width=True, key="tbl_saving_groq")

    st.subheader("💬 5. Feedback & Saran untuk Peserta")
    tab_feed_gemini, tab_feed_groq = st.tabs(["🤖 Saran GEMINI", "🦙 Saran GROQ LLAMA 3"])
    with tab_feed_gemini: st.data_editor(st.session_state.df_feedback_gemini, num_rows="dynamic", use_container_width=True, key="tbl_feed_gemini")
    with tab_feed_groq: st.data_editor(st.session_state.df_feedback_groq, num_rows="dynamic", use_container_width=True, key="tbl_feed_groq")

    with st.expander("📜 Lihat Transkrip Lengkap"):
        for entri in st.session_state.transkrip:
            st.markdown(f"**{entri['Peran']}**")
            st.text(entri["Laporan"])
            st.divider()

    # Logika Download Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        if not st.session_state.df_verifikasi.empty:
            st.session_state.df_verifikasi.to_excel(writer, sheet_name="1. Verifikasi Visual", index=False)
        if not st.session_state.df_alur_gemini.empty:
            st.session_state.df_alur_gemini.to_excel(writer, sheet_name="2. Alur Logika (Gemini)", index=False)
        if not st.session_state.df_alur_groq.empty:
            st.session_state.df_alur_groq.to_excel(writer, sheet_name="2. Alur Logika (Groq)", index=False)
        if not edited_rubrik_gemini.empty:
            edited_rubrik_gemini.to_excel(writer, sheet_name="3. Rubrik (Gemini)", index=False)
        if not edited_rubrik_groq.empty:
            edited_rubrik_groq.to_excel(writer, sheet_name="3. Rubrik (Groq)", index=False)
        if not st.session_state.df_saving_gemini.empty:
            st.session_state.df_saving_gemini.to_excel(writer, sheet_name="4. Saving (Gemini)", index=False)
        if not st.session_state.df_saving_groq.empty:
            st.session_state.df_saving_groq.to_excel(writer, sheet_name="4. Saving (Groq)", index=False)
        if not st.session_state.df_feedback_gemini.empty:
            st.session_state.df_feedback_gemini.to_excel(writer, sheet_name="5. Feedback (Gemini)", index=False)
        if not st.session_state.df_feedback_groq.empty:
            st.session_state.df_feedback_groq.to_excel(writer, sheet_name="5. Feedback (Groq)", index=False)

    excel_data = output.getvalue()

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Unduh Laporan Bandingan Lengkap (Excel)",
            data=excel_data,
            file_name=f"Laporan_Bandingan_{st.session_state.nama_file}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col2:
        if st.button("🔄 Unggah Dokumen Baru (Reset)"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
