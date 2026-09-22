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

# Injeksi CSS Kustom untuk Tema Merah-Putih Pastel
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

# Inisialisasi Gemini API Client dari Secrets
try:
  API_KEY = st.secrets["GEMINI_API_KEY"].strip()
  client = genai.Client(api_key=API_KEY)
except Exception as e:
  st.error(
      "Gagal memuat GEMINI_API_KEY. Pastikan di Secrets tertulis:"
      f' GEMINI_API_KEY = "..." | Error detail: {e}'
  )
  st.stop()

MODEL_ID = "gemini-1.5-flash"

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
  """Eksekusi panggilan API dengan jeda anti-limit dan log langsung ke UI."""
  for percobaan in range(maksimal_percobaan):
    try:
      log_ui.write(f"⏳ **{deskripsi_agen}:** Sedang menganalisis...")
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
          k in str(e).upper()
          for k in ["503", "429", "RESOURCE_EXHAUSTED", "UNAVAILABLE"]
      ):
        log_ui.write(
            f"⚠️ **{deskripsi_agen}:** Peladen sibuk/limit. Menunggu 20 detik..."
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
  """Memastikan ekstraksi JSON dari teks AI berjalan aman tanpa crash."""
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
      temp_path = f"temp_{uploaded_file.name}"
      try:
        status_box.write("📄 Membaca berkas PDF...")
        with open(temp_path, "wb") as f:
          f.write(uploaded_file.getbuffer())

        status_box.write("☁️ Mengunggah berkas ke Google AI Server...")
        gemini_file = client.files.upload(file=temp_path)

        # Memastikan status file aktif di server Google
        while gemini_file.state.name in ["PROCESSING", "PENDING"]:
          status_box.write(
              f"⏳ Menunggu verifikasi file di Google AI"
              f" ({gemini_file.state.name})..."
          )
          time.sleep(4)
          gemini_file = client.files.get(name=gemini_file.name)

        if gemini_file.state.name != "ACTIVE":
          raise Exception(
              "Berkas gagal diproses oleh Google AI. Status:"
              f" {gemini_file.state.name}"
          )

        # ------------------------------------------
        # AGEN 1: PENGEKSTRAK BUKTI
        # ------------------------------------------
        prompt_1 = """
        Kamu adalah Agen Pengekstrak Fakta Kaizen.
        Tugasmu: Ekstrak seluruh fakta objektif dari PDF meliputi:
        1. Masalah Utama (Kondisi Before)
        2. Solusi Utama (Kondisi After)
        3. Keberadaan Bukti Visual (Foto, Grafik, Diagram)
        4. Hasil Angka Nyata (Nilai Saving, Efisiensi Manpower, Ergonomi, Downtime, dll.)
        
        Berikan laporan berbasis fakta murni tanpa opini.
        """
        laporan_agen_1 = panggil_ai_dengan_retry(
            [gemini_file, prompt_1], "Pengekstrak Bukti [1/5]", status_box
        )

        # ------------------------------------------
        # AGEN 2: JAKSA PENILAI
        # ------------------------------------------
        prompt_2 = f"""
        Kamu adalah Agen Jaksa Penilai Kaizen yang sangat kritis dan skeptis.
        
        Fakta Kasus:
        {laporan_agen_1}
        
        Tugasmu:
        Cari seluruh kelemahan, celah, kecacatan metodologi, asumsi yang tidak terbukti, 
        kurangnya bukti visual, atau potensi manipulasi angka klaim dari paper ini.
        Berikan argumentasi penuntutan yang tajam dan objektif.
        """
        dakwaan_jaksa = panggil_ai_dengan_retry(
            prompt_2, "Jaksa Penilai [2/5]", status_box
        )

        # ------------------------------------------
        # AGEN 3: PENGACARA PEMBELA
        # ------------------------------------------
        prompt_3 = f"""
        Kamu adalah Agen Pengacara Pembela Kaizen.
        
        Fakta Kasus:
        {laporan_agen_1}
        
        Tuntutan Jaksa:
        {dakwaan_jaksa}
        
        Tugasmu:
        Bantah kritik Jaksa secara rasional. Temukan nilai tambah, usaha perbaikan, 
        serta dampak positif sekunder yang dicapai oleh tim Kaizen ini meskipun ada keterbatasan.
        """
        pembelaan_pengacara = panggil_ai_dengan_retry(
            prompt_3, "Pengacara Pembela [3/5]", status_box
        )

        # ------------------------------------------
        # AGEN 4: HAKIM AGUNG (21 POIN RUBRIK)
        # ------------------------------------------
        prompt_4 = f"""
        Kamu adalah Hakim Agung Evaluator Kaizen Manajerial yang berpengalaman dan obyektif.
        
        Fakta: {laporan_agen_1}
        Kritik Jaksa: {dakwaan_jaksa}
        Pembelaan: {pembelaan_pengacara}

        Evaluasi paper berdasarkan 21 Poin Rubrik berikut. Pilih skor HANYA dari nilai dalam kurung siku:
        1. 5G (G现场, G现物, G现实, G原理, G原则) [0, 1, 2]
        2. Losses Measurement [0, 1, 2]
        3. 5W1H Analysis [0, 1, 2]
        4. Visualisasi Masalah & Fenomena [0, 1, 2]
        5. Penetapan Target SMART [0, 2]
        6. Analisis Fishbone / 4M [0, 1, 2]
        7. Pemetaan Faktor 4M [0, 1, 2]
        8. Hubungan Akar Penyebab (Why-Why Analysis) [0, 1, 2]
        9. Bukti Verifikasi Akar Penyebab [0, 3, 5]
        10. Ketepatan Penetapan Root Cause [0, 1, 2]
        11. Action Plan & Penanggung Jawab (PIC) [0, 1, 2]
        12. Kelayakan Rencana Perbaikan [0, 1, 2]
        13. Form Usulan Perbaikan (FUP) [0, 3, 5]
        14. Eksekusi Pelaksanaan Action Plan [0, 1, 2]
        15. Dokumentasi Pelaksanaan (Before vs After) [0, 5, 8]
        16. Pencapaian Target Numerik [0, 1]
        17. Pengecekan Hasil & Evaluasi Dampak [0, 3, 5]
        18. Kelengkapan Dokumen Standardisasi [0, 3, 5]
        19. Validasi Standardisasi (Pengesahan) [0, 1, 2]
        20. Tindak Lanjut & Sosialisasi Standar Baru [0, 3, 5]
        21. Replikasi Improvement ke Line/Area Lain [0, 3, 5]

        Format Output WAJIB JSON ARRAY murni:
        [
          {{"no": 1, "kriteria": "5G", "skor": 2, "justifikasi": "alasan penjelasan singkat"}},
          ...lanjutkan sampai poin 21
        ]
        """
        raw_hakim = panggil_ai_dengan_retry(
            prompt_4, "Hakim Agung [4/5]", status_box
        )
        hasil_hakim_json = bersihkan_dan_parse_json(raw_hakim)

        # ------------------------------------------
        # AGEN 5: ANALIS IMPACT (SAVING)
        # ------------------------------------------
        prompt_5 = """
        Evaluasi dokumen Kaizen ini untuk 8 Kategori Saving/Impact berikut:
        1. Gas / Steam
        2. Material Balance
        3. Manpower
        4. Downtime
        5. Waktu Kerja
        6. Overtime
        7. Listrik
        8. Air

        Format Output WAJIB JSON ARRAY murni:
        [
          {"kategori": "Gas / Steam", "status": "YA", "keterangan": "penjelasan kuantitatif/kualitatif"},
          {"kategori": "Material Balance", "status": "TIDAK", "keterangan": "penjelasan singkat"},
          ...lanjutkan hingga 8 kategori
        ]
        """
        raw_analis = panggil_ai_dengan_retry(
            [gemini_file, prompt_5], "Analis Kesan [5/5]", status_box
        )
        hasil_saving_json = bersihkan_dan_parse_json(raw_analis)

        status_box.update(
            label="✅ Analisis Selesai!", state="complete", expanded=False
        )

        if os.path.exists(temp_path):
          os.remove(temp_path)

        # --- PEMROSESAN DATA TABEL ---
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
              for c in [
                  "no",
                  "kriteria",
                  "skor",
                  "status validasi",
                  "justifikasi",
              ]
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
        if os.path.exists(temp_path):
          os.remove(temp_path)
        status_box.update(label="❌ Terjadi Kesalahan", state="error")
        st.error(f"**Detail Pesan Ralat:** `{e}`")

# ==========================================
# 5. HASIL PENILAIAN & UNDUH EXCEL
# ==========================================
if st.session_state.proses_selesai:
  st.success(
      "Analisis AI selesai! Silakan periksa dan validasi tabel di bawah ini."
  )

  st.subheader("📝 1. Tabel Validasi Rubrik (21 Poin)")
  st.caption(
      "Baris bertanda **⚠️ VALIDASI MANUAL** dapat disesuaikan angkanya secara"
      " langsung pada tabel."
  )
  edited_rubrik = st.data_editor(
      st.session_state.df_rubrik, num_rows="dynamic", use_container_width=True
  )

  st.subheader("💰 2. Tabel Validasi Saving (8 Kategori)")
  edited_saving = st.data_editor(
      st.session_state.df_saving, num_rows="dynamic", use_container_width=True
  )

  # Pembuatan File Excel 3 Sheet
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
