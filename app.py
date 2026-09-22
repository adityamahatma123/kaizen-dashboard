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

# Inisialisasi API Key dari Secrets
try:
  API_KEY = st.secrets["GEMINI_API_KEY"].strip()
  client = genai.Client(api_key=API_KEY)
except Exception as e:
  st.error(f"Gagal memuat API Key dari Secrets. Detail: {e}")
  st.stop()

# Menggunakan model paling stabil untuk ekstraksi JSON saat ini
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
if "nama_file" not in st.session_state:
  st.session_state.nama_file = "Dokumen_Kaizen"


# ==========================================
# 3. FUNGSI MESIN AI & PARSER (ANTI-ERROR)
# ==========================================
def panggil_ai_dengan_retry(contents, deskripsi_agen, log_ui, maksimal_percobaan=3):
  """Eksekusi panggilan API dengan jeda anti-limit dan log langsung ke UI."""
  for percobaan in range(maksimal_percobaan):
    try:
      log_ui.write(f"⏳ **{deskripsi_agen}:** Sedang menganalisis...")
      response = client.models.generate_content(
          model=MODEL_ID, contents=contents
      )
      log_ui.write(
          f"✅ **{deskripsi_agen}:** Selesai! Pendinginan 12 detik"
          " (Anti-Limit)..."
      )
      time.sleep(12)
      return response.text if response and response.text else ""
    except Exception as e:
      if any(
          k in str(e).upper()
          for k in ["503", "429", "RESOURCE_EXHAUSTED", "UNAVAILABLE"]
      ):
        log_ui.write(
            f"⚠️ **{deskripsi_agen}:** Peladen sibuk. Menunggu 20 detik..."
        )
        time.sleep(20)
      else:
        if percobaan == maksimal_percobaan - 1:
          raise e
        log_ui.write(
            f"⚠️ **{deskripsi_agen}:** Mencoba ulang"
            f" ({percobaan + 2}/{maksimal_percobaan})..."
        )
        time.sleep(10)
  return ""


def bersihkan_dan_parse_json(teks_raw):
  """Fungsi pembaca JSON yang agresif mencari pola array tabel."""
  if not teks_raw:
    return []
  
  # 1. Bersihkan tanda format markdown secara paksa
  teks_bersih = re.sub(r"```json", "", teks_raw, flags=re.IGNORECASE)
  teks_bersih = re.sub(r"```", "", teks_bersih).strip()
  
  # 2. Coba parse sebagai objek JSON standar
  try:
    data = json.loads(teks_bersih)
    if isinstance(data, dict):
        # Jika AI membungkus dalam dictionary, misal {"hasil": [...]}, ekstrak list-nya
        for key, value in data.items():
            if isinstance(value, list):
                return value
        return [data]
    return data if isinstance(data, list) else []
  except json.JSONDecodeError:
    pass

  # 3. Jika gagal (ada teks ekstra), gunakan regex untuk memburu tanda kurung array [...]
  match = re.search(r'\[\s*\{.*?\}\s*\]', teks_raw, re.DOTALL)
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
      temp_path = f"temp_{st.session_state.nama_file}"
      try:
        status_box.write("📄 Membaca berkas PDF...")
        with open(temp_path, "wb") as f:
          f.write(uploaded_file.getbuffer())

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
          raise Exception(
              f"Berkas gagal diproses. Status: {gemini_file.state.name}"
          )

        # 1. Pengekstrak
        prompt_1 = (
            "Ekstrak fakta dari PDF: 1. Masalah Utama 2. Solusi 3. Bukti Visual"
            " 4. Hasil Angka Nyata (Saving)."
        )
        laporan_agen_1 = panggil_ai_dengan_retry(
            [gemini_file, prompt_1], "Pengekstrak Bukti [1/5]", status_box
        )

        # 2. Jaksa
        prompt_2 = (
            f"Fakta Kasus: {laporan_agen_1}\nTugasmu: Cari kelemahan, celah,"
            " kurangnya bukti, atau potensi manipulasi angka."
        )
        dakwaan_jaksa = panggil_ai_dengan_retry(
            prompt_2, "Jaksa Penilai [2/5]", status_box
        )

        # 3. Pembela
        prompt_3 = (
            f"Fakta: {laporan_agen_1}\nKritik: {dakwaan_jaksa}\nTugasmu: Bantah"
            " kritik Jaksa dan temukan nilai tambah."
        )
        pembelaan_pengacara = panggil_ai_dengan_retry(
            prompt_3, "Pengacara Pembela [3/5]", status_box
        )

        # 4. Hakim Agung 
        prompt_4 = f"""
        Fakta: {laporan_agen_1} | Kritik: {dakwaan_jaksa} | Pembelaan: {pembelaan_pengacara}
        Evaluasi 21 poin rubrik:
        1. 5G [0, 1, 2] | 2. Losses Measurement [0, 1, 2] | 3. 5W1H [0, 1, 2] | 4. Visualisasi [0, 1, 2] | 5. Target SMART [0, 2] | 6. Fishbone 4M [0, 1, 2] | 7. Pemetaan 4M [0, 1, 2] | 8. Hubungan Akar Penyebab [0, 1, 2] | 9. Bukti Akar Penyebab [0, 3, 5] | 10. Ketepatan Root Cause [0, 1, 2] | 11. Action Plan PIC [0, 1, 2] | 12. Rencana Perbaikan [0, 1, 2] | 13. Form Usulan Perbaikan [0, 3, 5] | 14. Pelaksanaan Action Plan [0, 1, 2] | 15. Dokumentasi Pelaksanaan [0, 5, 8] | 16. Pencapaian Target [0, 1] | 17. Pengecekan Hasil [0, 3, 5] | 18. Kelengkapan Standardisasi [0, 3, 5] | 19. Validasi Standardisasi [0, 1, 2] | 20. Tindak Lanjut Sosialisasi [0, 3, 5] | 21. Replikasi [0, 3, 5]
        
        KELUARKAN HANYA FORMAT JSON ARRAY: [{{"no": 1, "kriteria": "5G", "skor": 2, "justifikasi": "alasan"}}]
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
        if os.path.exists(temp_path):
          os.remove(temp_path)

        # --- PEMROSESAN DATA TABEL ---
        if hasil_hakim_json:
          poin_manual = [7, 10, 11, 18, 19, 21]
          for item in hasil_hakim_json:
            # Mencari nilai nomor kriteria meskipun AI menggunakan kapitalisasi berbeda
            nomor_kriteria = item.get("no", item.get("No", item.get("nomor", 0)))
            item["status validasi"] = (
                "⚠️ VALIDASI MANUAL"
                if nomor_kriteria in poin_manual
                else "OTOMATIS AI"
            )
          
          df_r = pd.DataFrame(hasil_hakim_json)
          # Menyeragamkan huruf kecil semua
          df_r.columns = df_r.columns.str.lower().str.strip()
          
          # Memperbaiki nama kolom jika AI salah memberikan nama
          df_r.rename(columns={"nomor": "no", "score": "skor", "nilai": "skor", "alasan": "justifikasi", "keterangan": "justifikasi"}, inplace=True)
          
          cols = [c for c in ["no", "kriteria", "skor", "status validasi", "justifikasi"] if c in df_r.columns]
          st.session_state.df_rubrik = df_r[cols] if cols else df_r
        else:
          # Fallback agar tabel tidak error 'empty' jika AI gagal
          st.session_state.df_rubrik = pd.DataFrame(columns=["no", "kriteria", "skor", "status validasi", "justifikasi"])

        if hasil_saving_json:
          df_s = pd.DataFrame(hasil_saving_json)
          df_s.columns = df_s.columns.str.lower().str.strip()
          st.session_state.df_saving = df_s
        else:
          st.session_state.df_saving = pd.DataFrame(columns=["kategori", "status", "keterangan"])

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
  st.success("Analisis AI selesai! Silakan periksa dan validasi tabel di bawah.")

  st.subheader("📝 1. Tabel Validasi Rubrik (21 Poin)")
  edited_rubrik = st.data_editor(
      st.session_state.df_rubrik, num_rows="dynamic", width="stretch", key="tabel_rubrik"
  )

  st.subheader("💰 2. Tabel Validasi Saving (8 Kategori)")
  edited_saving = st.data_editor(
      st.session_state.df_saving, num_rows="dynamic", width="stretch", key="tabel_saving"
  )

  output = io.BytesIO()
  with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    if not edited_rubrik.empty:
      edited_rubrik.to_excel(writer, sheet_name="Hasil Penilaian Rubrik", index=False)
    if not edited_saving.empty:
      edited_saving.to_excel(writer, sheet_name="Hasil Validasi Saving", index=False)
    if st.session_state.transkrip:
      pd.DataFrame(st.session_state.transkrip).to_excel(writer, sheet_name="Transkrip AI", index=False)

  excel_data = output.getvalue()

  col1, col2 = st.columns(2)
  with col1:
    st.download_button(
        label="📥 Unduh Laporan Lengkap (Excel 3 Sheet)",
        data=excel_data,
        file_name=f"Laporan_Kaizen_{st.session_state.nama_file}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
  with col2:
    if st.button("🔄 Unggah Dokumen Baru (Reset)"):
      st.session_state.proses_selesai = False
      st.session_state.df_rubrik = pd.DataFrame()
      st.session_state.df_saving = pd.DataFrame()
      st.session_state.transkrip = []
      st.session_state.nama_file = "Dokumen_Kaizen"
      st.rerun()
