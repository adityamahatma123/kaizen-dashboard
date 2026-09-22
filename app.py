import streamlit as st
import pandas as pd
import time
import json
import re
import os
from google import genai

# ==========================================
# 1. KONFIGURASI DASAR & API
# ==========================================
st.set_page_config(page_title="Dashboard Kaizen", layout="wide")
st.title("📊 Dashboard Validasi Kaizen - Manager")
st.write("Unggah dokumen Kaizen, biarkan AI menilai berdasarkan rubrik ketat, dan lakukan validasi akhir.")

try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    st.error("Gagal memuat API Key. Pastikan Anda sudah mengisinya di menu Settings > Secrets di Streamlit.")
    st.stop()

MODEL_ID = "gemini-3.5-flash"

# ==========================================
# 2. FUNGSI MESIN AI
# ==========================================
def panggil_ai_dengan_retry(contents, deskripsi_agen, maksimal_percobaan=3):
    """Fungsi eksekusi API dengan jeda otomatis untuk mencegah limit RPM."""
    for percobaan in range(maksimal_percobaan):
        try:
            response = client.models.generate_content(model=MODEL_ID, contents=contents)
            time.sleep(12) 
            return response.text
        except Exception as e:
            if any(k in str(e) for k in ["503", "429", "RESOURCE_EXHAUSTED"]):
                time.sleep(20)
            else:
                if percobaan == maksimal_percobaan - 1:
                    raise e
                time.sleep(10)

def bersihkan_dan_parse_json(teks_raw):
    """Fungsi memastikan output AI menjadi JSON array yang rapi."""
    try:
        teks_bersih = re.sub(r'```json\s*|\s*```', '', teks_raw).strip()
        return json.loads(teks_bersih)
    except Exception:
        match = re.search(r'(\[.*\]|\{.*\})', teks_raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return []

# ==========================================
# 3. ANTARMUKA UNGGAH & PROSES AI
# ==========================================
uploaded_file = st.file_uploader("Pilih file PDF Kaizen", type="pdf")

if uploaded_file is not None:
    st.success(f"File '{uploaded_file.name}' berhasil dimuat ke sistem.")
    
    if st.button("🚀 Mulai Penilaian AI (Proses ~2 Menit)"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # --- UPLOAD KE GOOGLE AI ---
            status_text.text("Menyiapkan dokumen di peladen Google...")
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            gemini_file = client.files.upload(file=temp_path)
            
            while gemini_file.state.name == "PROCESSING":
                time.sleep(3)
                gemini_file = client.files.get(name=gemini_file.name)
            progress_bar.progress(10)
            
            # --- AGEN 1: PENGEKSTRAK ---
            status_text.text("[1/5] Agen Pengekstrak sedang membaca dokumen...")
            prompt_1 = """Ekstrak fakta objektif dari PDF:
            1. Masalah utama (Before).
            2. Solusi utama (After).
            3. Bukti visual (Foto/grafik yang ada).
            4. Hasil angka nyata (Saving, waktu, ergonomi, dll)."""
            laporan_agen_1 = panggil_ai_dengan_retry([gemini_file, prompt_1], "Pengekstrak")
            progress_bar.progress(30)
            
            # --- AGEN 2: JAKSA ---
            status_text.text("[2/5] Agen Jaksa sedang menganalisis kelemahan...")
            prompt_2 = f"Fakta Kasus: {laporan_agen_1}\nTugasmu: Berikan kritik tajam mengenai kelemahan, potensi risiko, atau celah dari paper ini."
            dakwaan_jaksa = panggil_ai_dengan_retry(prompt_2, "Jaksa")
            progress_bar.progress(50)
            
            # --- AGEN 3: PEMBELA ---
            status_text.text("[3/5] Agen Pembela sedang menyusun argumen...")
            prompt_3 = f"Fakta: {laporan_agen_1}\nKritik: {dakwaan_jaksa}\nTugasmu: Bantah kritik Jaksa dan temukan nilai tambah dari perbaikan ini."
            pembelaan_pengacara = panggil_ai_dengan_retry(prompt_3, "Pembela")
            progress_bar.progress(70)
            
            # --- AGEN 4: HAKIM AGUNG (RUBRIK KETAT) ---
            status_text.text("[4/5] Hakim Agung sedang memberikan skor mutlak...")
            prompt_4 = f"""
            Kamu adalah 'Si Hakim Agung', evaluator Kaizen tingkat manajerial yang sangat KRITIS.
            Fakta: {laporan_agen_1}
            Kritik: {dakwaan_jaksa}
            Pembelaan: {pembelaan_pengacara}

            ATURAN MUTLAK (Pilih HANYA dari angka dalam kurung siku):
            1. 5G [0, 1, 2]
            2. Losses Measurement [0, 1, 2] -> Beri 0 jika kerugian tidak terukur numerik.
            3. 5W1H [0, 1, 2]
            4. Visualisasi (Fenomena) [0, 1, 2]
            5. Target SMART [0, 2]
            6. Fishbone Diagram / 4M [0, 1, 2] -> Beri 1 jika analisa dangkal.
            7. Pemetaan 4M [0, 1, 2]
            8. Hubungan Akar Penyebab [0, 1, 2] -> Beri 0 jika Why-Why tidak relevan/asumsi.
            9. Bukti Akar Penyebab [0, 3, 5] -> Beri 0 jika tidak ada foto/data empiris.
            10. Ketepatan Root Cause [0, 1, 2]
            11. Action Plan & PIC [0, 1, 2]
            12. Rencana Perbaikan [0, 1, 2]
            13. Form Usulan Perbaikan (FUP) [0, 3, 5] -> Beri 0 jika tidak ada bukti FUP.
            14. Pelaksanaan Action Plan [0, 1, 2]
            15. Dokumentasi Pelaksanaan [0, 5, 8] -> Beri 8 HANYA jika ada foto before-after valid.
            16. Pencapaian Target [0, 1]
            17. Pengecekan Hasil [0, 3, 5]
            18. Kelengkapan Standardisasi [0, 3, 5]
            19. Validasi Standardisasi [0, 1, 2] -> Beri 0 jika belum disahkan Section Head.
            20. Tindak Lanjut Sosialisasi [0, 3, 5]
            21. Replikasi Improvement [0, 3, 5] -> Beri 0 jika tidak direplikasi.

            KELUARKAN HANYA FORMAT JSON ARRAY:
            [
              {{"no": 1, "kriteria": "5G", "skor": 2, "justifikasi": "alasan singkat"}},
              ...lanjutkan hingga poin 21
            ]
            """
            raw_hakim = panggil_ai_dengan_retry(prompt_4, "Hakim Agung")
            hasil_hakim_json = bersihkan_dan_parse_json(raw_hakim)
            progress_bar.progress(90)
            
            # --- AGEN 5: ANALIS IMPACT ---
            status_text.text("[5/5] Analis Kesan sedang memeriksa kategori saving...")
            prompt_5 = """
            Evaluasi dokumen untuk 8 kategori impact: Gas/Steam, Material Balance, Manpower, Downtime, Waktu Kerja, Overtime, Listrik, Air.
            KELUARKAN HANYA FORMAT JSON ARRAY:
            [
              {"kategori": "Gas/Steam", "status": "TIDAK", "keterangan": "alasan singkat"},
              ...lanjutkan hingga 8 kategori
            ]
            """
            raw_analis = panggil_ai_dengan_retry([gemini_file, prompt_5], "Analis Kesan")
            hasil_saving_json = bersihkan_dan_parse_json(raw_analis)
            
            progress_bar.progress(100)
            status_text.text("✅ Analisis Selesai!")
            os.remove(temp_path)
            
            # ==========================================
            # 4. MENAMPILKAN HASIL UNTUK DIVALIDASI
            # ==========================================
            st.divider()
            
            # A. TAMPILAN TABEL RUBRIK (Bisa Diedit)
            if hasil_hakim_json:
                poin_manual = [7, 10, 11, 18, 19, 21]
                for item in hasil_hakim_json:
                    item['Status'] = "⚠️ VALIDASI MANUAL" if item.get('no') in poin_manual else "OTOMATIS AI"
                
                df_rubrik = pd.DataFrame(hasil_hakim_json)
                if 'no' in df_rubrik.columns:
                    df_rubrik = df_rubrik[['no', 'kriteria', 'skor', 'Status', 'justifikasi']]
                
                st.subheader("📝 Tabel Validasi Rubrik (21 Poin)")
                st.info("Klik ganda (double-click) pada kolom 'skor' untuk mengubah nilai secara manual sesuai kondisi lapangan.")
                edited_df_rubrik = st.data_editor(df_rubrik, num_rows="dynamic", use_container_width=True)
                
                # Fitur Unduh CSV Rubrik
                csv_rubrik = edited_df_rubrik.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Unduh Hasil Rubrik (CSV)", data=csv_rubrik, file_name=f"Rubrik_{uploaded_file.name}.csv", mime="text/csv")
            
            st.write("")
            
            # B. TAMPILAN TABEL SAVING
            if hasil_saving_json:
                df_saving = pd.DataFrame(hasil_saving_json)
                st.subheader("💰 Tabel Kategori Saving")
                edited_df_saving = st.data_editor(df_saving, num_rows="dynamic", use_container_width=True)
                
                csv_saving = edited_df_saving.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Unduh Hasil Saving (CSV)", data=csv_saving, file_name=f"Saving_{uploaded_file.name}.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses AI: {e}")
