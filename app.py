import streamlit as st
import pandas as pd
import time
import json
import re
import os
import io
from google import genai

# ==========================================
# 1. KONFIGURASI DASAR & API
# ==========================================
st.set_page_config(page_title="Dashboard Kaizen", layout="wide")
st.title("📊 Dashboard Validasi Kaizen - Manager")
st.write("Unggah dokumen Kaizen, biarkan AI menilai, dan unduh laporan Excel multi-sheet.")

try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    st.error("Gagal memuat API Key. Pastikan Anda sudah mengisinya di menu Settings > Secrets.")
    st.stop()

MODEL_ID = "gemini-3.5-flash"

# ==========================================
# 2. INISIALISASI MEMORI WEB (SESSION STATE)
# ==========================================
# Ini mencegah web tereset setelah tombol unduh ditekan
if "proses_selesai" not in st.session_state:
    st.session_state.proses_selesai = False
if "df_rubrik" not in st.session_state:
    st.session_state.df_rubrik = pd.DataFrame()
if "df_saving" not in st.session_state:
    st.session_state.df_saving = pd.DataFrame()
if "transkrip" not in st.session_state:
    st.session_state.transkrip = []

# ==========================================
# 3. FUNGSI MESIN AI
# ==========================================
def panggil_ai_dengan_retry(contents, deskripsi_agen, maksimal_percobaan=3):
    for percobaan in range(maksimal_percobaan):
        try:
            response = client.models.generate_content(model=MODEL_ID, contents=contents)
            time.sleep(12) 
            # PENAHAN ERROR: Pastikan selalu mengembalikan string, walaupun respons aslinya kosong (None)
            return response.text if response.text else ""
        except Exception as e:
            if any(k in str(e) for k in ["503", "429", "RESOURCE_EXHAUSTED"]):
                time.sleep(20)
            else:
                if percobaan == maksimal_percobaan - 1:
                    raise e
                time.sleep(10)

def bersihkan_dan_parse_json(teks_raw):
    # PENAHAN ERROR: Jika teks_raw kosong atau None, langsung kembalikan list kosong tanpa memproses RegEx
    if not teks_raw:
        return []
        
    try:
        teks_bersih = re.sub(r'```json\s*|\s*```', '', teks_raw).strip()
        return json.loads(teks_bersih)
    except Exception:
        match = re.search(r'(\[.*\]|\{.*\})', teks_raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return []

# ==========================================
# 4. ANTARMUKA UNGGAH & PROSES AI
# ==========================================
uploaded_file = st.file_uploader("Pilih file PDF Kaizen", type="pdf")

# Jika tombol analisis ditekan, proses AI berjalan
if uploaded_file is not None and not st.session_state.proses_selesai:
    if st.button("🚀 Mulai Penilaian AI (Proses ~2 Menit)"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Menyiapkan dokumen di peladen Google...")
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            gemini_file = client.files.upload(file=temp_path)
            while gemini_file.state.name == "PROCESSING":
                time.sleep(3)
                gemini_file = client.files.get(name=gemini_file.name)
            progress_bar.progress(10)
            
            # --- Rantai Agen AI ---
            status_text.text("[1/5] Agen Pengekstrak sedang membaca dokumen...")
            prompt_1 = "Ekstrak fakta objektif dari PDF: 1. Masalah utama (Before). 2. Solusi utama (After). 3. Bukti visual. 4. Hasil angka nyata (Saving)."
            laporan_agen_1 = panggil_ai_dengan_retry([gemini_file, prompt_1], "Pengekstrak")
            progress_bar.progress(30)
            
            status_text.text("[2/5] Agen Jaksa sedang menganalisis kelemahan...")
            prompt_2 = f"Fakta Kasus: {laporan_agen_1}\nTugasmu: Berikan kritik tajam mengenai kelemahan, potensi risiko, atau celah paper."
            dakwaan_jaksa = panggil_ai_dengan_retry(prompt_2, "Jaksa")
            progress_bar.progress(50)
            
            status_text.text("[3/5] Agen Pembela sedang menyusun argumen...")
            prompt_3 = f"Fakta: {laporan_agen_1}\nKritik: {dakwaan_jaksa}\nTugasmu: Bantah kritik Jaksa dan temukan nilai tambah."
            pembelaan_pengacara = panggil_ai_dengan_retry(prompt_3, "Pembela")
            progress_bar.progress(70)
            
            status_text.text("[4/5] Hakim Agung sedang memberikan skor mutlak...")
            prompt_4 = f"""
            Evaluasi 21 poin rubrik Kaizen. Kamu adalah penilai yang ketat.
            Fakta: {laporan_agen_1}
            Kritik: {dakwaan_jaksa}
            Pembelaan: {pembelaan_pengacara}

            1. 5G [0, 1, 2] | 2. Losses Measurement [0, 1, 2] | 3. 5W1H [0, 1, 2] | 4. Visualisasi [0, 1, 2] | 5. Target SMART [0, 2] | 6. Fishbone 4M [0, 1, 2] | 7. Pemetaan 4M [0, 1, 2] | 8. Hubungan Akar Penyebab [0, 1, 2] | 9. Bukti Akar Penyebab [0, 3, 5] | 10. Ketepatan Root Cause [0, 1, 2] | 11. Action Plan PIC [0, 1, 2] | 12. Rencana Perbaikan [0, 1, 2] | 13. Form Usulan Perbaikan [0, 3, 5] | 14. Pelaksanaan Action Plan [0, 1, 2] | 15. Dokumentasi Pelaksanaan [0, 5, 8] | 16. Pencapaian Target [0, 1] | 17. Pengecekan Hasil [0, 3, 5] | 18. Kelengkapan Standardisasi [0, 3, 5] | 19. Validasi Standardisasi [0, 1, 2] | 20. Tindak Lanjut Sosialisasi [0, 3, 5] | 21. Replikasi [0, 3, 5]
            
            KELUARKAN HANYA FORMAT JSON ARRAY: [{{"no": 1, "kriteria": "5G", "skor": 2, "justifikasi": "alasan"}}]
            """
            raw_hakim = panggil_ai_dengan_retry(prompt_4, "Hakim Agung")
            hasil_hakim_json = bersihkan_dan_parse_json(raw_hakim)
            progress_bar.progress(90)
            
            status_text.text("[5/5] Analis Kesan sedang memeriksa kategori saving...")
            prompt_5 = "Evaluasi 8 kategori impact: Gas/Steam, Material Balance, Manpower, Downtime, Waktu Kerja, Overtime, Listrik, Air.\nKELUARKAN JSON ARRAY: [{\"kategori\": \"Air\", \"status\": \"TIDAK\", \"keterangan\": \"alasan\"}]"
            raw_analis = panggil_ai_dengan_retry([gemini_file, prompt_5], "Analis Kesan")
            hasil_saving_json = bersihkan_dan_parse_json(raw_analis)
            
            progress_bar.progress(100)
            status_text.text("✅ Analisis Selesai!")
            os.remove(temp_path)
            
            # --- Menyimpan Hasil ke Memori Sesi ---
            if hasil_hakim_json:
                poin_manual = [7, 10, 11, 18, 19, 21]
                for item in hasil_hakim_json:
                    item['Status Validasi'] = "⚠️ VALIDASI MANUAL" if item.get('no') in poin_manual else "OTOMATIS AI"
                
                df_r = pd.DataFrame(hasil_hakim_json)
                if 'no' in df_r.columns:
                    st.session_state.df_rubrik = df_r[['no', 'kriteria', 'skor', 'Status Validasi', 'justifikasi']]
            
            if hasil_saving_json:
                st.session_state.df_saving = pd.DataFrame(hasil_saving_json)
            
            st.session_state.transkrip = [
                {"Peranan": "Ejen Pengekstrak", "Laporan": laporan_agen_1},
                {"Peranan": "Ejen Jaksa", "Laporan": dakwaan_jaksa},
                {"Peranan": "Ejen Pembela", "Laporan": pembelaan_pengacara},
                {"Peranan": "Ejen Hakim", "Laporan": raw_hakim},
                {"Peranan": "Ejen Analis", "Laporan": raw_analis}
            ]
            
            st.session_state.proses_selesai = True
            st.rerun() # Memuat ulang layar agar tabel muncul

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

# ==========================================
# 5. MENAMPILKAN HASIL & UNDUH EXCEL
# ==========================================
if st.session_state.proses_selesai:
    st.success("Data berhasil diproses! Silakan validasi tabel di bawah ini.")
    
    st.subheader("📝 1. Tabel Validasi Rubrik")
    edited_rubrik = st.data_editor(st.session_state.df_rubrik, num_rows="dynamic", use_container_width=True)
    
    st.subheader("💰 2. Tabel Validasi Saving")
    edited_saving = st.data_editor(st.session_state.df_saving, num_rows="dynamic", use_container_width=True)
    
    # Membuat File Excel Multi-Sheet di dalam memori
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        edited_rubrik.to_excel(writer, sheet_name='Hasil Penilaian Rubrik', index=False)
        edited_saving.to_excel(writer, sheet_name='Hasil Validasi Saving', index=False)
        pd.DataFrame(st.session_state.transkrip).to_excel(writer, sheet_name='Transkrip AI', index=False)
    
    excel_data = output.getvalue()
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Unduh Laporan Lengkap (Excel)",
            data=excel_data,
            file_name=f"Laporan_Kaizen_{uploaded_file.name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with col2:
        if st.button("🔄 Unggah Dokumen Baru (Reset)"):
            # Membersihkan memori untuk memulai dari awal
            st.session_state.proses_selesai = False
            st.session_state.df_rubrik = pd.DataFrame()
            st.session_state.df_saving = pd.DataFrame()
            st.session_state.transkrip = []
            st.rerun()
