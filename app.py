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

# ==========================================
# 1. KONFIGURASI HALAMAN & TEMA (UI/UX)
# ==========================================
st.set_page_config(
    page_title="Portal Validasi Kaizen", page_icon="🏢", layout="wide"
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #B03A2E; text-align: center; font-family: 'Segoe UI', sans-serif; font-weight: 700; }
    .stButton>button { background-color: #E07A5F; color: white; border-radius: 6px; border: none; padding: 0.5rem 1rem; font-weight: 600; }
    .stButton>button:hover { background-color: #C85A3F; color: white; }
    .manual-badge { background-color: #FCE8E6; color: #B03A2E; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🏢 Portal Validasi Kaizen")
st.markdown(
    "<p style='text-align: center; color: #555; font-size: 1.1rem;'>Unggah"
    " dokumen evaluasi, biarkan AI bekerja secara objektif dan konsisten,"
    " lalu lakukan validasi akhir secara manual pada poin-poin kritikal.</p>",
    unsafe_allow_html=True,
)
st.divider()

# Inisialisasi API Key dari Secrets
try:
  API_KEY = st.secrets["GEMINI_API_KEY"].strip()
  client = genai.Client(api_key=API_KEY)
except Exception as e:
  st.error(f"Gagal memuat API Key dari Secrets. Detail: {e}")
  st.stop()

# Model yang digunakan. "Lite" dipilih karena free tier-nya jauh lebih
# longgar (15 request/menit) dibanding gemini-3.5-flash biasa (cuma 5
# request/menit) — penting karena alur ini melakukan 5 panggilan berurutan
# per dokumen. Kalau nanti sudah mengaktifkan billing dan ingin kualitas
# penalaran lebih dalam, ganti kembali ke "gemini-3.5-flash".
MODEL_ID = "gemini-3.5-flash-lite"

# Parameter untuk memaksimalkan KONSISTENSI hasil antar-run.
# CATATAN PENTING: temperature=0 + top_k=1 TIDAK direkomendasikan untuk
# model Gemini 3.x (termasuk gemini-3.5-flash) karena model ini punya mode
# "thinking" internal — kombinasi itu bisa membuat model terjebak di proses
# berpikir tanpa pernah mengeluarkan jawaban akhir (response.text jadi
# kosong). Gunakan thinking_level + seed sebagai gantinya.
# thinking_level="medium" dipakai (bukan "low") karena penilaian sekarang
# butuh penalaran mendalam: memverifikasi rantai logika 5 Whys, ketepatan
# kategori 4M pada fishbone, dan konsistensi target-vs-hasil — bukan cuma
# mengecek ada/tidaknya elemen.
GENERATION_CONFIG_TEXT = types.GenerateContentConfig(
    seed=42,
    thinking_config=types.ThinkingConfig(thinking_level="medium"),
)

GENERATION_CONFIG_JSON = types.GenerateContentConfig(
    seed=42,
    thinking_config=types.ThinkingConfig(thinking_level="medium"),
    response_mime_type="application/json",
)

# Poin rubrik yang WAJIB divalidasi manusia (hybrid) karena butuh konteks
# aktual di lapangan yang tidak bisa diketahui AI dari dokumen saja.
POIN_VALIDASI_MANUAL = {
    7: "Pemetaan 4M — perlu verifikasi kesesuaian dengan kondisi mesin/area aktual",
    10: "Ketepatan Root Cause — perlu justifikasi teknis dari asesor lapangan",
    11: "Action Plan PIC — perlu konfirmasi PIC & jadwal riil",
    18: "Kelengkapan Standardisasi — perlu cek dokumen fisik/SOP terbaru",
    19: "Validasi Standardisasi — perlu verifikasi implementasi di lapangan",
    21: "Replikasi — perlu konfirmasi area lain yang benar-benar direplikasi",
}

# Rubrik 21 poin PERSIS dari dokumen manual "Rubrik Penilaian Kaizen 2026 —
# Bagian II". Deskripsi tiap tingkat skor disertakan lengkap (bukan cuma
# nama kriteria) supaya AI menilai berdasarkan definisi asli, bukan
# tebakan generik "ada/tidak ada".
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
13. Form Usulan Perbaikan (FUP) — 0: Tidak ada pendaftaran FUP | 3: Sudah didaftarkan FUP namun belum dapat approval | 5: Sudah didaftarkan FUP dan sudah dapat approval (dapat dibuktikan), ATAU action plan memang tidak memerlukan FUP

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

# 14 kategori impact PERSIS dari dokumen manual "Bagian I — Form Penilaian".
KATEGORI_IMPACT_14 = [
    "Gas / Steam",
    "Material Balance",
    "Manpower",
    "Downtime",
    "Waktu / Proses Kerja",
    "Overtime",
    "Listrik",
    "Air",
    "Stock Accuracy",
    "Inventory / Material Value",
    "DOI",
    "Quality",
    "Safety & Environment",
    "SOC & HTA",
]


# ==========================================
# 2. INISIALISASI MEMORI SESI (SESSION STATE)
# ==========================================
if "proses_selesai" not in st.session_state:
  st.session_state.proses_selesai = False
if "df_rubrik" not in st.session_state:
  st.session_state.df_rubrik = pd.DataFrame()
if "df_alur_logika" not in st.session_state:
  st.session_state.df_alur_logika = pd.DataFrame()
if "df_verifikasi" not in st.session_state:
  st.session_state.df_verifikasi = pd.DataFrame()
if "df_saving" not in st.session_state:
  st.session_state.df_saving = pd.DataFrame()
if "transkrip" not in st.session_state:
  st.session_state.transkrip = []
if "nama_file" not in st.session_state:
  st.session_state.nama_file = "Dokumen_Kaizen"


# ==========================================
# 3. FUNGSI MESIN AI & PARSER (ANTI-ERROR)
# ==========================================
def panggil_ai_dengan_retry(
    contents,
    deskripsi_agen,
    log_ui,
    config=None,
    maksimal_percobaan=3,
):
  """Eksekusi panggilan API dengan jeda anti-limit dan log langsung ke UI."""
  config = config or GENERATION_CONFIG_TEXT
  for percobaan in range(maksimal_percobaan):
    try:
      log_ui.write(f"⏳ **{deskripsi_agen}:** Sedang menganalisis...")
      response = client.models.generate_content(
          model=MODEL_ID, contents=contents, config=config
      )
      teks_hasil = response.text if response and response.text else ""
      if not teks_hasil:
        finish_reason = None
        try:
          finish_reason = response.candidates[0].finish_reason
        except Exception:
          pass
        log_ui.write(
            f"❗ **{deskripsi_agen}:** Jawaban KOSONG dari model"
            f" (finish_reason: {finish_reason}). Mencoba ulang..."
        )
        if percobaan == maksimal_percobaan - 1:
          log_ui.write(
              f"❌ **{deskripsi_agen}:** Tetap kosong setelah"
              f" {maksimal_percobaan}x percobaan."
          )
          return ""
        time.sleep(8)
        continue
      log_ui.write(
          f"✅ **{deskripsi_agen}:** Selesai! Pendinginan 15 detik"
          " (menjaga di bawah limit 5 request/menit free tier)..."
      )
      time.sleep(15)
      return teks_hasil
    except Exception as e:
      pesan_error_asli = str(e)
      pesan_error_upper = pesan_error_asli.upper()
      if any(
          k in pesan_error_upper
          for k in ["503", "429", "RESOURCE_EXHAUSTED", "UNAVAILABLE"]
      ):
        log_ui.write(
            f"⚠️ **{deskripsi_agen}:** Kena limit rate (kemungkinan 5"
            f" request/menit free tier terlampaui). Detail:"
            f" `{pesan_error_asli[:300]}`. Menunggu 65 detik agar masuk"
            " jendela menit berikutnya..."
        )
        time.sleep(65)
      else:
        if percobaan == maksimal_percobaan - 1:
          raise
        log_ui.write(
            f"⚠️ **{deskripsi_agen}:** Error: `{pesan_error_asli[:300]}`."
            f" Mencoba ulang ({percobaan + 2}/{maksimal_percobaan})..."
        )
        time.sleep(10)
  log_ui.write(
      f"❌ **{deskripsi_agen}:** Menyerah setelah {maksimal_percobaan}x"
      " percobaan (kemungkinan kuota/rate limit API habis — cek Google AI"
      " Studio / billing akun Gemini-mu)."
  )
  return ""


def bersihkan_dan_parse_json(teks_raw):
  """Fungsi pembaca JSON yang agresif mencari pola array tabel."""
  if not teks_raw:
    return []

  teks_bersih = re.sub(r"```json", "", teks_raw, flags=re.IGNORECASE)
  teks_bersih = re.sub(r"```", "", teks_bersih).strip()

  try:
    data = json.loads(teks_bersih)
    if isinstance(data, dict):
      for _key, value in data.items():
        if isinstance(value, list):
          return value
      return [data]
    return data if isinstance(data, list) else []
  except json.JSONDecodeError:
    pass

  match = re.search(r"\[\s*\{.*?\}\s*\]", teks_raw, re.DOTALL)
  if match:
    try:
      return json.loads(match.group(0))
    except Exception:
      pass

  return []


# ==========================================
# 4. ALUR UNGGAH & EKSEKUSI MULTI-AGENT
# ==========================================
uploaded_file = st.file_uploader("Pilih file PDF Kaizen", type="pdf")

if uploaded_file is not None and not st.session_state.proses_selesai:
  if st.button("🚀 Mulai Penilaian AI"):
    st.session_state.nama_file = uploaded_file.name

    with st.status(
        "🤖 AI Multi-Agent sedang bekerja...", expanded=True
    ) as status_box:
      # Gunakan folder temp resmi sistem (lebih aman untuk cloud/multi-user)
      suffix = os.path.splitext(uploaded_file.name)[1] or ".pdf"
      temp_path = None
      gemini_file = None
      try:
        status_box.write("📄 Membaca berkas PDF...")
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as tmp:
          tmp.write(uploaded_file.getbuffer())
          temp_path = tmp.name

        status_box.write("☁️ Mengunggah berkas ke Google AI Server...")
        gemini_file = client.files.upload(file=temp_path)

        while gemini_file.state.name in ["PROCESSING", "PENDING"]:
          status_box.write(
              f"⏳ Menunggu verifikasi file di Google AI"
              f" ({gemini_file.state.name})..."
          )
          time.sleep(4)
          gemini_file = client.files.get(name=gemini_file.name)

        if gemini_file.state.name != "ACTIVE":
          raise RuntimeError(
              f"Berkas gagal diproses. Status: {gemini_file.state.name}"
          )

        # --- 1. Pengekstrak Fakta ---
        prompt_1 = (
            "Kamu adalah auditor dokumen Kaizen yang teliti dan hanya"
            " melaporkan fakta yang benar-benar tertulis/tervisualisasi di"
            " dokumen, tanpa asumsi atau tambahan opini. Ekstrak SETIAP"
            " elemen berikut secara VERBATIM/detail (kutip isi aslinya,"
            " jangan diringkas berlebihan), karena akan dipakai untuk"
            " analisis koherensi logika, bukan sekadar cek ada/tidak:\n\n"
            "1. MASALAH UTAMA: kondisi awal, data pendukung, 5W1H lengkap"
            " (What/Where/When/Who/Why/How), evidence 5G.\n"
            "2. TARGET AWAL (SMART): kutip persis angka/kalimat target yang"
            " ditetapkan di awal dokumen.\n"
            "3. FISHBONE DIAGRAM: untuk SETIAP cabang/duri yang ada, sebutkan"
            " (a) kategori 4M yang dipakai dokumen (Man/Method/"
            "Machine/Material), (b) isi penyebab yang dituliskan di cabang"
            " itu. Buat sebagai daftar, contoh: 'Man: operator kurang"
            " terlatih', 'Machine: mesin sering aus'.\n"
            "4. ANALISIS 5 WHYS: kutip SETIAP baris why secara berurutan"
            " dan lengkap (why 1 sampai why terakhir) apa adanya, jangan"
            " diringkas. Sebutkan juga apa root cause final yang diklaim"
            " dokumen.\n"
            "5. ACTION PLAN & PIC: daftar rencana perbaikan beserta"
            " penanggung jawab (PIC) dan status FUP (Form Usulan"
            " Perbaikan) bila disebutkan.\n"
            "6. IMPLEMENTASI: bukti pelaksanaan (dokumentasi, foto"
            " before/after, laporan trial).\n"
            "7. HASIL AKHIR/PENCAPAIAN: kutip persis angka hasil akhir yang"
            " dilaporkan (termasuk saving), dan periode pengukurannya.\n"
            "8. STANDARDISASI: dokumen IK/SOP/OPL/CILT/PM/Centerline yang"
            " dibuat, status validasi/approval, bukti sosialisasi"
            " (absensi), dan bukti replikasi ke area/mesin lain.\n\n"
            "Jika suatu elemen tidak ditemukan di dokumen, nyatakan dengan"
            " jelas 'TIDAK DITEMUKAN' — jangan mengarang."
        )
        laporan_agen_1 = panggil_ai_dengan_retry(
            [gemini_file, prompt_1], "Pengekstrak Bukti [1/7]", status_box
        )

        # --- 2. Verifikator Bukti Visual & Kelayakan (multimodal) ---
        # Agen ini WAJIB punya akses langsung ke file PDF (bukan cuma teks
        # ringkasan) karena tugasnya melihat foto/gambar secara visual dan
        # menilai kelayakan dasar proyek — dua hal yang tidak bisa dicek
        # dari ringkasan teks Pengekstrak saja.
        prompt_verifikasi = """Kamu adalah verifikator senior yang SANGAT KRITIS terhadap kualitas dasar submission Kaizen. Banyak peserta kompetisi ini belum paham konsep PDCA dengan baik, dan beberapa submission bahkan BUKAN merupakan proyek improvement sama sekali (misal: cuma laporan rutin, pengadaan barang tanpa problem-solving, atau aktivitas maintenance biasa yang dibungkus format Kaizen). Tugasmu membongkar ini dengan membaca LANGSUNG dokumen PDF (termasuk semua foto/gambar/diagram di dalamnya), bukan cuma ringkasan.

Lakukan 3 pemeriksaan berikut:

## A. GATE CHECK — Kelayakan sebagai Proyek Improvement
Apakah dokumen ini benar-benar proyek continuous improvement yang valid? Tanda-tanda TIDAK LAYAK: tidak ada kondisi awal/masalah yang didefinisikan dengan jelas, tidak ada perubahan before-after yang nyata, tidak ada analisis akar masalah sama sekali (langsung lompat ke solusi), atau isinya sebenarnya laporan administratif/aktivitas rutin yang dipaksakan ke format Kaizen. Beri verdict: "LAYAK" (jelas proyek improvement yang sah), "PERLU PERHATIAN" (ada keraguan, perlu ditinjau juri), atau "TIDAK LAYAK" (bukan proyek improvement).

## B. KEBENARAN SEMANTIK 5W1H
Untuk MASING-MASING elemen (What, Where, When, Who, Why, How — atau elemen serupa yang dipakai dokumen), periksa apakah ISI yang dituliskan benar-benar menjawab pertanyaan elemen itu, bukan cuma ada teks di kolomnya. Contoh kesalahan yang harus ditangkap: isi kolom "How" sebenarnya menjelaskan "Where" (lokasi), atau isi "Which"/kolom lain tertukar dengan elemen lain. Tandai tiap elemen SESUAI atau TERTUKAR/TIDAK SESUAI dengan penjelasan spesifik.

## C. AUDIT FOTO & BUKTI VISUAL
Untuk SETIAP foto/gambar/diagram penting yang kamu lihat di dokumen (terutama foto before/after, dan diagram fishbone/flow), deskripsikan singkat apa yang benar-benar terlihat di foto itu, lalu bandingkan dengan klaim teks di sekitarnya. Tandai SESUAI kalau foto benar-benar mendukung klaim (misal foto "before" menunjukkan kondisi rusak/bermasalah yang dideskripsikan, foto "after" menunjukkan perbaikan yang sama), atau MERAGUKAN kalau foto tidak jelas mendukung klaim, tidak relevan, tampak diambil dari konteks lain, atau foto before/after terlihat identik/tidak menunjukkan perubahan nyata.

Keluarkan HANYA JSON array valid (satu array datar berisi semua temuan A+B+C, dibedakan lewat field "kategori"), dengan skema persis:
[{"kategori": "GATE CHECK", "item": "Kelayakan Proyek Improvement", "status": "LAYAK", "catatan": "alasan spesifik merujuk isi dokumen"}, {"kategori": "5W1H", "item": "How", "status": "TERTUKAR/TIDAK SESUAI", "catatan": "isi kolom How sebenarnya menjelaskan lokasi (Where), bukan metode"}, {"kategori": "FOTO", "item": "Foto halaman 8 (before)", "status": "SESUAI", "catatan": "menunjukkan kondisi mesin sesuai deskripsi masalah"}]"""
        raw_verifikasi = panggil_ai_dengan_retry(
            [gemini_file, prompt_verifikasi],
            "Verifikator Bukti & Kelayakan [2/7]",
            status_box,
            config=GENERATION_CONFIG_JSON,
        )
        hasil_verifikasi_json = bersihkan_dan_parse_json(raw_verifikasi)

        # --- 3. Pelacak Alur Logika (Traceability/Golden Thread PDCA) ---
        prompt_alur = f"""Kamu adalah auditor senior metodologi Kaizen/PDCA yang bertugas menelusuri "benang merah" (golden thread) logika perbaikan — apakah tiap tahap benar-benar tersambung secara MASUK AKAL secara teknis/operasional ke tahap sebelumnya, bukan cuma nyambung secara kata-kata di permukaan.

Gunakan pengetahuan umum troubleshooting industri sebagai patokan kewajaran hubungan sebab-akibat. Contoh pola yang MASUK AKAL: "mesin macet" -> kenapa? "bearing aus" -> kenapa? "kurang pelumasan" -> kenapa? "tidak ada jadwal preventive maintenance". Contoh loncatan yang TIDAK MASUK AKAL: "mesin macet" tiba-tiba dijawab "operator kurang training" tanpa ada penjelasan penghubung antara dua hal itu.

DATA HASIL EKSTRAKSI DOKUMEN:
{laporan_agen_1}

HASIL VERIFIKASI KEBENARAN 5W1H & FOTO (pakai ini sebagai bahan tambahan, terutama untuk titik 1 di bawah — kalau ada elemen 5W1H yang TERTUKAR, itu otomatis bikin problem statement TIDAK bisa dianggap KONSISTEN):
{raw_verifikasi}

Telusuri dan evaluasi titik-titik sambungan berikut SATU PER SATU:
1. "5W1H ke Problem Statement": apakah rangkuman masalah mencerminkan semua elemen 5W1H yang dilaporkan DENGAN BENAR (bukan cuma lengkap, tapi juga tidak ada elemen yang tertukar/salah tempat sesuai hasil verifikasi di atas), tidak ada yang hilang/melenceng?
2. "Problem Statement ke Kepala Ikan Fishbone": apakah efek/masalah utama di kepala ikan fishbone SAMA dengan problem statement di atas, atau melenceng ke masalah lain?
3. "Kategori 4M pada Fishbone": untuk SETIAP cabang, apakah penyebabnya masuk akal ditempatkan di kategori 4M itu? Sebutkan SPESIFIK cabang mana yang salah kategori kalau ada (misal: soal alat/mesin dimasukkan ke Man).
4. "Rantai Why-Why": untuk SETIAP pasangan why berurutan, apakah why berikutnya benar-benar penyebab TEKNIS LANGSUNG dari why sebelumnya (bukan cuma berkaitan tema)? Sebutkan link why-ke-berapa yang lemah/meloncat kalau ada.
5. "Root Cause vs Fishbone": apakah root cause akhir dari why-why itu memang salah satu cabang di fishbone, bukan topik baru yang tidak pernah disebut?
6. "Root Cause ke Action Plan": apakah action plan yang diajukan menyasar ROOT CAUSE AKHIR (why paling dalam), bukan cuma menambal gejala di why tingkat awal?
7. "Action Plan ke Standardisasi": apakah dokumen standar (SOP/IK/OPL/dst) yang dibuat memang relevan dan mengunci action plan itu supaya tidak terulang?
8. "Target Awal ke Hasil Akhir": apakah angka target di awal benar-benar dijawab hasil akhir yang dilaporkan, tanpa pergeseran target yang tidak dijelaskan?

Untuk tiap titik, beri verdict SALAH SATU dari: "KONSISTEN" (jelas dan masuk akal), "LEMAH" (ada tapi kurang detail/agak dipaksakan), atau "TIDAK KONSISTEN" (ada loncatan logika/tidak nyambung/tidak ditemukan).

Keluarkan HANYA JSON array valid dengan skema persis:
[{{"no": 1, "tahap": "5W1H ke Problem Statement", "verdict": "KONSISTEN", "temuan": "penjelasan spesifik merujuk isi dokumen"}}]"""
        raw_alur_logika = panggil_ai_dengan_retry(
            prompt_alur,
            "Pelacak Alur Logika [3/7]",
            status_box,
            config=GENERATION_CONFIG_JSON,
        )
        hasil_alur_json = bersihkan_dan_parse_json(raw_alur_logika)

        # --- 3. Jaksa (Kritik) ---
        prompt_2 = (
            "Kamu berperan sebagai Jaksa yang skeptis dan sangat teliti"
            " dalam audit Kaizen. Tugasmu mencari kelemahan KOHERENSI dan"
            " LOGIKA, bukan cuma kelengkapan administratif, berdasarkan"
            " fakta yang ada (jangan mengarang tuduhan).\n\n"
            f"Fakta Kasus:\n{laporan_agen_1}\n\n"
            "HASIL AUDIT ALUR LOGIKA (dari Pelacak Alur Logika, jadikan"
            f" bahan utama kritikmu):\n{raw_alur_logika}\n\n"
            "Periksa dan pertanyakan secara spesifik, dengan mengacu ke"
            " hasil audit alur logika di atas:\n"
            "- Titik mana saja yang berstatus 'LEMAH' atau 'TIDAK"
            " KONSISTEN' — jelaskan kenapa itu masalah serius untuk"
            " kredibilitas penilaian.\n"
            "- Kelemahan bukti, celah antara masalah dan solusi, kurangnya"
            " data pendukung, atau potensi manipulasi angka saving.\n\n"
            "Sertakan alasan yang merujuk ke fakta di atas untuk tiap"
            " temuan."
        )
        dakwaan_jaksa = panggil_ai_dengan_retry(
            prompt_2, "Jaksa Penilai [4/7]", status_box
        )

        # --- 5. Pembela ---
        prompt_3 = (
            "Kamu berperan sebagai Pengacara Pembela dalam audit Kaizen."
            " Tugasmu membela HANYA berdasarkan fakta yang tersedia, bukan"
            " asumsi baik yang tidak berdasar.\n\n"
            f"Fakta:\n{laporan_agen_1}\n\nKritik Jaksa:\n{dakwaan_jaksa}\n\n"
            "Bantah kritik yang tidak berdasar dan soroti nilai tambah yang"
            " sudah terbukti dari fakta di atas."
        )
        pembelaan_pengacara = panggil_ai_dengan_retry(
            prompt_3, "Pengacara Pembela [5/7]", status_box
        )

        # --- 6. Hakim Agung (Skoring 21 Poin) ---
        prompt_4 = f"""Kamu adalah Hakim Agung penilaian Kaizen yang wajib bersikap objektif, konsisten, dan KRITIS TERHADAP ISI — bukan cuma mengecek "ada/tidak ada elemen", tapi memverifikasi apakah isinya benar secara logika, tepat kategorinya, dan nyambung alur PDCA-nya.

ATURAN PENILAIAN:
- Beri skor SESUAI pilihan yang tersedia per kriteria (jangan beri skor di luar pilihan yang tercantum di rubrik).
- Ikuti PERSIS deskripsi tiap tingkat skor di rubrik di bawah — jangan menebak sendiri artinya.
- Untuk kriteria 1, 2, 3, dan 4 (5G, Losses Measurement, 5W1H, Visualisasi), jadikan HASIL VERIFIKASI 5W1H & FOTO di bawah sebagai bahan utama — kalau ada elemen 5W1H yang "TERTUKAR/TIDAK SESUAI" atau foto yang "MERAGUKAN", skor kriteria terkait WAJIB ikut diturunkan meski secara sepintas terlihat "ada isinya".
- Untuk kriteria 6, 7, 8, 10, dan 16, jadikan HASIL AUDIT ALUR LOGIKA di bawah sebagai bahan utama penilaian — kalau audit menyatakan "TIDAK KONSISTEN" atau "LEMAH" pada titik yang relevan, skor kriteria itu WAJIB ikut diturunkan dan justifikasi WAJIB menyebutkan temuan spesifik dari audit tersebut.
- Kalau GATE CHECK di hasil verifikasi menyatakan "TIDAK LAYAK" (bukan proyek improvement), sebutkan ini secara eksplisit di justifikasi kriteria 1-5 (tahap Plan) karena ini mempengaruhi validitas keseluruhan submission — tapi tetap beri skor per kriteria sesuai bukti yang ada (jangan otomatis nol semua tanpa dasar).
- Justifikasi WAJIB spesifik dan merujuk isi konkret dari dokumen (kutip singkat bagian relevan bila perlu), bukan opini umum seperti "sudah lengkap" atau "ada bukti".
- Bersikap ketat: skor tinggi hanya untuk bukti yang benar-benar kuat, lengkap, DAN koheren secara logika.

RUBRIK LENGKAP (deskripsi tiap tingkat skor):
{RUBRIK_21_POIN_DETAIL}

FAKTA HASIL EKSTRAKSI DOKUMEN:
{laporan_agen_1}

HASIL VERIFIKASI KELAYAKAN, 5W1H & FOTO (pakai untuk kriteria 1-5 dan sebagai konteks kelayakan keseluruhan):
{raw_verifikasi}

HASIL AUDIT ALUR LOGIKA (golden thread PDCA — pakai ini terutama untuk kriteria 6, 7, 8, 10, 16):
{raw_alur_logika}

KRITIK JAKSA (pertimbangkan temuan ini dalam penilaian):
{dakwaan_jaksa}

PEMBELAAN:
{pembelaan_pengacara}

Keluarkan HANYA JSON array valid, tanpa teks lain, dengan skema persis (justifikasi harus spesifik dan merujuk isi dokumen serta hasil audit alur logika/verifikasi, minimal 1-2 kalimat menjelaskan MENGAPA skor itu diberikan berdasarkan analisis koherensi, bukan cuma "ada evidence"):
[{{"no": 1, "kriteria": "5G", "skor": 2, "justifikasi": "alasan spesifik merujuk isi dokumen dan analisis koherensi"}}]"""
        raw_hakim = panggil_ai_dengan_retry(
            prompt_4,
            "Hakim Agung [6/7]",
            status_box,
            config=GENERATION_CONFIG_JSON,
        )
        hasil_hakim_json = bersihkan_dan_parse_json(raw_hakim)

        # --- 7. Analis Dampak ---
        daftar_kategori_str = ", ".join(KATEGORI_IMPACT_14)
        prompt_5 = (
            "Kamu adalah Analis Impact yang menilai dampak operasional dari"
            " dokumen Kaizen ini secara objektif berdasarkan bukti"
            " tertulis saja.\n\n"
            f"Evaluasi {len(KATEGORI_IMPACT_14)} kategori impact berikut:"
            f" {daftar_kategori_str}.\n"
            "Untuk setiap kategori, status HARUS salah satu dari: 'IYA'"
            " (ada dampak terbukti dengan data/keterangan jelas di"
            " dokumen), 'TIDAK' (tidak ada dampak/tidak disebutkan sama"
            " sekali), atau 'TIDAK YAKIN' (disinggung tapi tidak jelas/"
            " tidak ada data pendukung yang cukup).\n\n"
            "SELAIN itu, tentukan juga jenis saving berdasarkan dokumen:"
            " apakah termasuk 'Hard Saving' (saving real >100 juta rupiah,"
            " terkait penurunan pemakaian gas/listrik/air/uji riksa/"
            " pembelian material), 'Virtual Saving atau Cost Avoidance'"
            " (saving tidak real, terkait material balance/stock akurasi/"
            "customer complain), 'Keduanya', atau 'Tidak Ada' — tambahkan"
            " sebagai satu entri terpisah dengan kategori bernilai"
            " 'Jenis Saving'.\n\n"
            "Keluarkan HANYA JSON array valid dengan skema persis:\n"
            '[{"kategori": "Air", "status": "TIDAK", "keterangan":'
            ' "alasan singkat merujuk dokumen"}]'
        )
        raw_analis = panggil_ai_dengan_retry(
            [gemini_file, prompt_5],
            "Analis Dampak [7/7]",
            status_box,
            config=GENERATION_CONFIG_JSON,
        )
        hasil_saving_json = bersihkan_dan_parse_json(raw_analis)

        ada_yang_gagal = not all([
            laporan_agen_1,
            raw_verifikasi,
            raw_alur_logika,
            dakwaan_jaksa,
            pembelaan_pengacara,
            raw_hakim,
            raw_analis,
        ])
        status_box.update(
            label=(
                "⚠️ Selesai dengan beberapa agen gagal — lihat detail di"
                " bawah"
                if ada_yang_gagal
                else "✅ Analisis Selesai!"
            ),
            state="complete" if not ada_yang_gagal else "error",
            expanded=ada_yang_gagal,
        )

        # --- PEMROSESAN DATA TABEL ---
        if hasil_verifikasi_json:
          df_v = pd.DataFrame(hasil_verifikasi_json)
          df_v.columns = df_v.columns.str.lower().str.strip()
          st.session_state.df_verifikasi = df_v
        else:
          st.session_state.df_verifikasi = pd.DataFrame(
              columns=["kategori", "item", "status", "catatan"]
          )

        if hasil_alur_json:
          df_a = pd.DataFrame(hasil_alur_json)
          df_a.columns = df_a.columns.str.lower().str.strip()
          st.session_state.df_alur_logika = df_a
        else:
          st.session_state.df_alur_logika = pd.DataFrame(
              columns=["no", "tahap", "verdict", "temuan"]
          )

        if hasil_hakim_json:
          for item in hasil_hakim_json:
            nomor_kriteria = item.get(
                "no", item.get("No", item.get("nomor", 0))
            )
            try:
              nomor_kriteria = int(nomor_kriteria)
            except (TypeError, ValueError):
              nomor_kriteria = 0
            item["status validasi"] = (
                "⚠️ VALIDASI MANUAL"
                if nomor_kriteria in POIN_VALIDASI_MANUAL
                else "OTOMATIS AI"
            )
            item["skor_ai"] = item.get("skor", item.get("score", None))
            item["skor_final"] = item["skor_ai"]
            item["catatan_validator"] = ""

          df_r = pd.DataFrame(hasil_hakim_json)
          df_r.columns = df_r.columns.str.lower().str.strip()
          df_r.rename(
              columns={
                  "nomor": "no",
                  "score": "skor",
                  "nilai": "skor",
                  "alasan": "justifikasi",
                  "keterangan": "justifikasi",
              },
              inplace=True,
          )

          kolom_urutan = [
              "no",
              "kriteria",
              "status validasi",
              "skor_ai",
              "skor_final",
              "justifikasi",
              "catatan_validator",
          ]
          cols = [c for c in kolom_urutan if c in df_r.columns]
          sisa = [c for c in df_r.columns if c not in cols]
          st.session_state.df_rubrik = df_r[cols + sisa]
        else:
          st.session_state.df_rubrik = pd.DataFrame(
              columns=[
                  "no",
                  "kriteria",
                  "status validasi",
                  "skor_ai",
                  "skor_final",
                  "justifikasi",
                  "catatan_validator",
              ]
          )

        if hasil_saving_json:
          df_s = pd.DataFrame(hasil_saving_json)
          df_s.columns = df_s.columns.str.lower().str.strip()
          st.session_state.df_saving = df_s
        else:
          st.session_state.df_saving = pd.DataFrame(
              columns=["kategori", "status", "keterangan"]
          )

        st.session_state.transkrip = [
            {"Peran": "Agen Pengekstrak", "Laporan": laporan_agen_1},
            {
                "Peran": "Agen Verifikator Bukti & Kelayakan",
                "Laporan": raw_verifikasi,
            },
            {"Peran": "Agen Pelacak Alur Logika", "Laporan": raw_alur_logika},
            {"Peran": "Agen Jaksa Penilai", "Laporan": dakwaan_jaksa},
            {
                "Peran": "Agen Pengacara Pembela",
                "Laporan": pembelaan_pengacara,
            },
            {"Peran": "Agen Hakim Agung", "Laporan": raw_hakim},
            {"Peran": "Agen Analis Dampak", "Laporan": raw_analis},
        ]

        st.session_state.proses_selesai = True
        st.rerun()

      except Exception as e:
        status_box.update(label="❌ Terjadi Kesalahan", state="error")
        st.error(f"**Pesan Error:** `{e}`")
        with st.expander("🔍 Detail teknis (traceback lengkap)"):
          st.code(traceback.format_exc())
      finally:
        if temp_path and os.path.exists(temp_path):
          try:
            os.remove(temp_path)
          except OSError:
            pass
        if gemini_file is not None:
          try:
            client.files.delete(name=gemini_file.name)
          except Exception:
            pass

# ==========================================
# 5. HASIL PENILAIAN & UNDUH EXCEL
# ==========================================
if st.session_state.proses_selesai:
  st.success(
      "Analisis AI selesai! Silakan periksa dan validasi tabel di bawah —"
      " poin bertanda ⚠️ WAJIB divalidasi manual sebelum diunduh."
  )

  # Peringatan permanen jika ada agen yang gagal menghasilkan jawaban,
  # supaya tidak perlu bongkar transkrip / log status untuk tahu masalahnya.
  agen_kosong = [
      entri["Peran"]
      for entri in st.session_state.transkrip
      if not str(entri.get("Laporan", "")).strip()
  ]
  if agen_kosong:
    st.warning(
        "⚠️ Agen berikut TIDAK menghasilkan jawaban (kemungkinan diblokir"
        " safety filter, kehabisan kuota, atau macet di proses berpikir"
        " model):\n\n"
        + "\n".join(f"- {nama}" for nama in agen_kosong)
        + "\n\nTabel di bawah mungkin kosong/tidak lengkap akibat ini."
        " Coba jalankan ulang, atau cek log saat proses berjalan untuk"
        " detail `finish_reason`."
    )

  # Ambil verdict gate check untuk ditampilkan sebagai banner besar paling
  # atas, supaya juri bisa triase cepat: layak dinilai detail atau tidak.
  verdict_gate = None
  alasan_gate = ""
  if not st.session_state.df_verifikasi.empty:
    baris_gate = st.session_state.df_verifikasi[
        st.session_state.df_verifikasi.get("kategori", pd.Series(dtype=str))
        .astype(str)
        .str.upper()
        == "GATE CHECK"
    ]
    if not baris_gate.empty:
      verdict_gate = str(baris_gate.iloc[0].get("status", "")).upper()
      alasan_gate = str(baris_gate.iloc[0].get("catatan", ""))

  if verdict_gate == "TIDAK LAYAK":
    st.error(
        f"🚫 **GATE CHECK: TIDAK LAYAK sebagai proyek improvement.**"
        f" {alasan_gate}\n\nTinjau ulang apakah submission ini memang"
        " layak dinilai sebagai Kaizen, sebelum melanjutkan ke skor"
        " detail di bawah."
    )
  elif verdict_gate == "PERLU PERHATIAN":
    st.warning(
        f"⚠️ **GATE CHECK: PERLU PERHATIAN.** {alasan_gate}\n\nBukan"
        " berarti otomatis gugur, tapi juri disarankan meninjau langsung"
        " bagian ini sebelum finalisasi skor."
    )
  elif verdict_gate == "LAYAK":
    st.success(f"✅ **GATE CHECK: LAYAK sebagai proyek improvement.** {alasan_gate}")

  st.subheader("🔍 -1. Verifikasi Kelayakan, Kebenaran 5W1H & Bukti Foto")
  st.caption(
      "Hasil dari agen yang membaca LANGSUNG file PDF (termasuk foto/"
      "gambar di dalamnya) — bukan cuma ringkasan teks. Kolom **status**"
      " 'TERTUKAR/TIDAK SESUAI' atau 'MERAGUKAN' berarti ada isi yang"
      " keliru ditempatkan atau foto yang tidak benar-benar mendukung"
      " klaim di sekitarnya."
  )
  edited_verifikasi = st.data_editor(
      st.session_state.df_verifikasi,
      num_rows="dynamic",
      use_container_width=True,
      key="tabel_verifikasi",
  )

  st.subheader("🔗 0. Audit Alur Logika PDCA (Golden Thread)")
  st.caption(
      "Menelusuri apakah tiap tahap (5W1H → Problem Statement → Kepala"
      " Ikan → Kategori 4M → Rantai Why-Why → Root Cause → Action Plan →"
      " Standardisasi → Target vs Hasil) benar-benar tersambung logis,"
      " bukan cuma sama-sama ada. **LEMAH/TIDAK KONSISTEN** di sini jadi"
      " dasar penurunan skor kriteria 6, 7, 8, 10, dan 16 di tabel rubrik"
      " bawah."
  )
  edited_alur = st.data_editor(
      st.session_state.df_alur_logika,
      num_rows="dynamic",
      use_container_width=True,
      key="tabel_alur_logika",
  )

  st.subheader("📝 1. Tabel Validasi Rubrik (21 Poin)")
  st.caption(
      "Kolom **skor_ai** adalah skor asli dari AI (jangan diubah, sebagai"
      " jejak audit). Isi/ubah **skor_final** dan **catatan_validator**"
      " untuk poin yang bertanda ⚠️ VALIDASI MANUAL."
  )
  edited_rubrik = st.data_editor(
      st.session_state.df_rubrik,
      num_rows="dynamic",
      use_container_width=True,
      key="tabel_rubrik",
      column_config={
          "skor_ai": st.column_config.NumberColumn(disabled=True),
      },
  )

  total_ai = pd.to_numeric(
      edited_rubrik.get("skor_ai"), errors="coerce"
  ).sum()
  total_final = pd.to_numeric(
      edited_rubrik.get("skor_final"), errors="coerce"
  ).sum()
  col_a, col_b = st.columns(2)
  col_a.metric("Total Skor AI (awal)", f"{total_ai:.0f}")
  col_b.metric("Total Skor Final (setelah validasi)", f"{total_final:.0f}")

  st.subheader("💰 2. Tabel Validasi Impact & Saving (14 Kategori)")
  edited_saving = st.data_editor(
      st.session_state.df_saving,
      num_rows="dynamic",
      use_container_width=True,
      key="tabel_saving",
  )

  with st.expander("📜 Lihat Transkrip Lengkap Multi-Agent"):
    for entri in st.session_state.transkrip:
      st.markdown(f"**{entri['Peran']}**")
      st.text(entri["Laporan"])
      st.divider()

  output = io.BytesIO()
  with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    if not edited_verifikasi.empty:
      edited_verifikasi.to_excel(
          writer, sheet_name="Verifikasi 5W1H & Foto", index=False
      )
    if not edited_alur.empty:
      edited_alur.to_excel(
          writer, sheet_name="Audit Alur Logika", index=False
      )
    if not edited_rubrik.empty:
      edited_rubrik.to_excel(
          writer, sheet_name="Hasil Penilaian Rubrik", index=False
      )
    if not edited_saving.empty:
      edited_saving.to_excel(
          writer, sheet_name="Hasil Validasi Saving", index=False
      )
    if st.session_state.transkrip:
      pd.DataFrame(st.session_state.transkrip).to_excel(
          writer, sheet_name="Transkrip AI", index=False
      )

  excel_data = output.getvalue()

  col1, col2 = st.columns(2)
  with col1:
    st.download_button(
        label="📥 Unduh Laporan Lengkap (Excel 5 Sheet)",
        data=excel_data,
        file_name=f"Laporan_Kaizen_{st.session_state.nama_file}.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument"
            ".spreadsheetml.sheet"
        ),
    )
  with col2:
    if st.button("🔄 Unggah Dokumen Baru (Reset)"):
      st.session_state.proses_selesai = False
      st.session_state.df_verifikasi = pd.DataFrame()
      st.session_state.df_alur_logika = pd.DataFrame()
      st.session_state.df_rubrik = pd.DataFrame()
      st.session_state.df_saving = pd.DataFrame()
      st.session_state.transkrip = []
      st.session_state.nama_file = "Dokumen_Kaizen"
      st.rerun()
