import io
import json
import os
import re
import time
from google import genai
import pandas as pd
import streamlit as st

# ==========================================
# 1. KONFIGURASI HALAMAN & TEMA (UI/UX)
# ==========================================
st.set_page_config(
    page_title="Portal Validasi Kaizen", page_icon="🏢", layout="wide"
)

# Injeksi CSS Kustom untuk Tema Merah-Putih Pastel & Tampilan Portal
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    h1 {
        color: #B03A2E; 
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
    }
    
    .stButton>button {
        background-color: #E07A5F;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #C85A3F;
        color: white;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🏢 Portal Validasi Kaizen")
st.markdown(
    "<p style='text-align: center; color: #555; font-size: 1.1rem;'>Unggah"
    " dokumen evaluasi, biarkan AI bekerja secara objektif, dan lakukan"
    " validasi akhir.</p>",
    unsafe_allow_html=True,
)
st.divider()

# Sambungan ke Gemini API via Secrets
try:
  API_KEY = st.secrets["GEMINI_API_KEY"]
  client = genai.Client(api_key=API_KEY)
except Exception:
  st.error(
      "Gagal memuat GEMINI_API_KEY. Pastikan kunci rahasia sudah diset di menu"
      " Settings > Secrets pada Streamlit Cloud."
  )
  st.stop()

MODEL_ID = "gemini-3.5-flash"

# ==========================================
# 2. INISIALISASI MEMORI SESI (SESSION STATE)
# ==========================================
if "proses_selesai" not in st.session_state:
  st.session_state.proses_selesai = False
if "df_rubrik" not in st.session_state:
  st.session_state.df_rubrik = pd.DataFrame()
if "df_saving" not in st.session_state:
  st.session_state.df_saving = pd.DataFrame()
if "transkrip" not in st.session_state:
  st.session_state.transkrip = []


# ==========================================
# 3. FUNGSI MESIN AI & PARSER (ANTI-ERROR)
# ==========================================
def panggil_ai_dengan_retry(
    contents, deskripsi_agen, log_ui, maksimal_percobaan=3
):
  """Panggilan API Gemini dengan jeda anti-limit dan log langsung ke antarmuka."""
  for percobaan in range(maksimal_percobaan):
    try:
      log_ui.write(f"⏳ **{deskripsi_agen}:** Sedang menganalisis dokumen...")
      response = client.models.generate_content(
          model=MODEL_ID, contents=contents
      )

      log_ui.write(
          f"✅ **{deskripsi_agen}:** Selesai! Menunggu pendinginan 12 detik"
          " (anti-limit RPM)..."
      )
      time.sleep(12)

      return response.text if response and response.text else ""
    except Exception as e:
      if any(
          k in str(e) for k in ["503", "429", "RESOURCE_EXHAUSTED", "UNAVAILABLE"]
      ):
        log_ui.write(
            f"⚠️ **{deskripsi_agen}:** Peladen Google sibuk. Menunggu 20"
            " detik..."
        )
        time.sleep(20)
      else:
        if percobaan == maksimal_percobaan - 1:
          raise e
        log_ui.write(
            f"⚠️ **{deskripsi_agen}:** Mencoba ulang (Percobaan"
            f" {percobaan + 2}/{maksimal_percobaan})..."
        )
        time.sleep(10)
  return ""


def bersihkan_dan_parse_json(teks_raw):
  """Memastikan output AI diekstrak menjadi JSON array yang sah (kebal NoneType)."""
  if not teks_raw:
    return []
  try:
    teks_bersih = re.sub(r"```json\s*|\s*```", "", teks_raw).strip()
    return json.loads(teks_bersih)
  except Exception:
    match = re.search(r"(\[.*\]|\{.*\})", teks_raw, re.DOTALL)
    if match:
      try:
        return json.loads(match.group(1))
      except Exception:
        return []
    return []


# ==========================================
# 4. ALUR UNGGAH & EKSEKUSI MULTI-AGENT
# ==========================================
uploaded_file = st.file_uploader("Pilih file PDF Kaizen", type="pdf")

if uploaded_file is not None and not st.session_state.proses_selesai:
  if st.button("🚀 Mulai Penilaian AI"):

    with st.status(
        "🤖 AI Multi-Agent sedang bekerja...", expanded=True
    ) as status_box:
      try:
        status_box.write("📄 Mengunggah dokumen ke peladen Google...")
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
          f.write(uploaded_file.getbuffer())

        gemini_file = client.files.upload(file=temp_path)
        while gemini_file.state.name == "PROCESSING":
          status_box.write("⏳ Memproses berkas PDF di sistem Google...")
          time.sleep(3)
          gemini_file = client.files.get(name=gemini_file.name)

        # 1. Pengekstrak
        prompt_1 = (
            "Ekstrak fakta objektif dari PDF: 1. Masalah utama (Before). 2."
            " Solusi utama (After). 3. Bukti visual. 4. Hasil angka nyata"
            " (saving)."
        )
        laporan_agen_1 = panggil_ai_dengan_retry(
            [gemini_file, prompt_1], "Pengekstrak Bukti [1/5]", status_box
        )

        # 2. Jaksa
        prompt_2 = (
            f"Fakta Kasus: {laporan_agen_1}\nTugasmu: Berikan kritik tajam"
            " mengenai kelemahan, potensi risiko, atau celah dari paper ini."
        )
        dakwaan_jaksa = panggil_ai_dengan_retry(
            prompt_2, "Jaksa Penilai [2/5]", status_box
        )

        # 3. Pembela
        prompt_3 = (
            f"Fakta: {laporan_agen_1}\nKritik: {dakwaan_jaksa}\nTugasmu: Bantah"
            " kritik Jaksa dan temukan nilai tambah dari perbaikan ini."
        )
        pembelaan_pengacara = panggil_ai_dengan_retry(
            prompt_3, "Pengacara Pembela [3/5]", status_box
        )

        # 4. Hakim Agung (Rubrik Ketat)
        prompt_4 = f"""
                Evaluasi 21 poin rubrik Kaizen. Kamu adalah penilai yang ketat dan objektif.
                Fakta: {laporan_agen_1}
                Kritik: {dakwaan_jaksa}
                Pembelaan: {pembelaan_pengacara}

                1. 5G [0, 1, 2] | 2. Losses Measurement [0, 1, 2] | 3. 5W1H [0, 1, 2] | 4. Visualisasi [0, 1, 2] | 5. Target SMART [0, 2] | 6. Fishbone 4M [0, 1, 2] | 7. Pemetaan 4M [0, 1, 2] | 8. Hubungan Akar Penyebab [0, 1, 2] | 9. Bukti Akar Penyebab [0, 3, 5] | 10. Ketepatan Root Cause [0, 1, 2] | 11. Action Plan PIC [0, 1, 2] | 12. Rencana Perbaikan [0, 1, 2] | 13. Form Usulan Perbaikan [0, 3, 5] | 14. Pelaksanaan Action Plan [0, 1, 2] | 15. Dokumentasi Pelaksanaan [0, 5, 8] | 16. Pencapaian Target [0, 1] | 17. Pengecekan Hasil [0, 3, 5] | 18. Kelengkapan Standardisasi [0, 3, 5] | 19. Validasi Standardisasi [0, 1, 2] | 20. Tindak Lanjut Sosialisasi [0, 3, 5] | 21. Replikasi [0, 3, 5]
                
                KELUARKAN HANYA FORMAT JSON ARRAY: [{"no": 1, "kriteria": "5G", "skor": 2, "justifikasi": "alasan"}]
                """
        raw_hakim = panggil_ai_dengan_retry(
            prompt_4, "Hakim Agung [4/5]", status_box
        )
        hasil_hakim_json = bersihkan_dan_parse_json(raw_hakim)

        # 5. Analis Impact
        prompt_5 = (
            "Evaluasi 8 kategori impact: Gas/Steam, Material Balance,"
            " Manpower, Downtime, Waktu Kerja, Overtime, Listrik, Air.\nKELUARKAN"
            ' JSON ARRAY: [{"kategori": "Air", "status": "TIDAK", "keterangan":'
            ' "alasan"}]'
        )
        raw_analis = panggil_ai_dengan_retry(
            [gemini_file, prompt_5], "Analis Kesan [5/5]", status_box
        )
        hasil_saving_json = bersihkan_dan_parse_json(raw_analis)

        status_box.update(
            label="✅ Analisis Selesai!", state="complete", expanded=False
        )
        os.remove(temp_path)

        # --- PEMROSESAN DATA UNTUK TABEL & MEMORI ---
        if hasil_hakim_json:
          poin_manual = [7, 10, 11, 18, 19, 21]
          for item in hasil_hakim_json:
            nomor_kriteria = item.get("no", item.get("No"))
            item["status validasi"] = (
                "⚠️ VALIDASI MANUAL"
                if nomor_kriteria in poin_manual
                else "OTOMATIS AI"
            )

          df_r = pd.DataFrame(hasil_hakim_json)
          df_r.columns = df_r.columns.str.lower()

          cols = [
              c
              for c in ["no", "kriteria", "skor", "status validasi", "justifikasi"]
              if c in df_r.columns
          ]
          st.session_state.df_rubrik = df_r[cols]

        if hasil_saving_json:
          df_s = pd.DataFrame(hasil_saving_json)
          df_s.columns = df_s.columns.str.lower()
          st.session_state.df_saving = df_s

        st.session_state.transkrip = [
            {"Peranan": "Ejen Pengekstrak", "Laporan": laporan_agen_1},
            {"Peranan": "Ejen Jaksa Penilai", "Laporan": dakwaan_jaksa},
            {"Peranan": "Ejen Pengacara Pembela", "Laporan": pembelaan_pengacara},
            {"Peranan": "Ejen Hakim Agung", "Laporan": raw_hakim},
            {"Peranan": "Ejen Analis Kesan", "Laporan": raw_analis},
        ]

        st.session_state.proses_selesai = True
        st.rerun()

      except Exception as e:
        status_box.update(label="❌ Terjadi Kesalahan", state="error")
        st.error(f"Rincian kesalahan: {e}")

# ==========================================
# 5. HASIL PENILAIAN & UNDUH EXCEL MULTI-SHEET
# ==========================================
if st.session_state.proses_selesai:
  st.success("Analisis AI selesai! Silakan periksa dan validasi tabel di bawah.")

  st.subheader("📝 1. Tabel Validasi Rubrik (21 Poin)")
  st.caption(
      "Baris dengan status **⚠️ VALIDASI MANUAL** dapat disesuaikan nilainya"
      " secara langsung pada tabel."
  )
  edited_rubrik = st.data_editor(
      st.session_state.df_rubrik, num_rows="dynamic", use_container_width=True
  )

  st.subheader("💰 2. Tabel Validasi Saving (8 Kategori)")
  edited_saving = st.data_editor(
      st.session_state.df_saving, num_rows="dynamic", use_container_width=True
  )

  # Menyiapkan File Excel 3 Sheet di Memori
  output = io.BytesIO()
  with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    edited_rubrik.to_excel(
        writer, sheet_name="Hasil Penilaian Rubrik", index=False
    )
    edited_saving.to_excel(
        writer, sheet_name="Hasil Validasi Saving", index=False
    )
    pd.DataFrame(st.session_state.transkrip).to_excel(
        writer, sheet_name="Transkrip AI", index=False
    )

  excel_data = output.getvalue()

  col1, col2 = st.columns(2)
  with col1:
    st.download_button(
        label="📥 Unduh Laporan Lengkap (Excel 3 Sheet)",
        data=excel_data,
        file_name=f"Laporan_Kaizen_{uploaded_file.name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
  with col2:
    if st.button("🔄 Unggah Dokumen Baru (Reset)"):
      st.session_state.proses_selesai = False
      st.session_state.df_rubrik = pd.DataFrame()
      st.session_state.df_saving = pd.DataFrame()
      st.session_state.transkrip = []
      st.rerun()
