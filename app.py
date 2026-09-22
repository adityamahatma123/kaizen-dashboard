import streamlit as st
import pandas as pd

# Mengatur tata letak halaman web
st.set_page_config(page_title="Dashboard Kaizen", layout="wide")

st.title("📊 Dashboard Validasi Kaizen - Manager")
st.write("Unggah dokumen Kaizen, biarkan AI menilai, dan validasi hasilnya di sini.")

# 1. Fitur Upload PDF
uploaded_file = st.file_uploader("Pilih file PDF Kaizen", type="pdf")

if uploaded_file is not None:
    st.success(f"File '{uploaded_file.name}' berhasil diunggah!")
    
    # Tombol untuk memicu AI (simulasi sementara)
    if st.button("Mulai Penilaian AI"):
        with st.spinner("AI sedang memproses dokumen... (Simulasi)"):
            
            # --- Di sinilah nanti kode Gemini 5 Agen kita masukkan ---
            
            # Data simulasi untuk menampilkan tabel
            data_dummy = {
                "No": [1, 2, 19, 21],
                "Kriteria": ["5G", "Losses", "Validasi Standardisasi", "Replikasi"],
                "Skor AI": [2, 1, 0, 0],
                "Status": ["OTOMATIS AI", "OTOMATIS AI", "⚠️ VALIDASI MANUAL", "⚠️ VALIDASI MANUAL"],
                "Justifikasi": ["Bukti foto ada.", "Target jelas.", "Belum ada TTD Sec Head.", "Tidak ada replikasi."]
            }
            df = pd.DataFrame(data_dummy)
            
            st.write("### 📝 Hasil Penilaian Rubrik (Silakan Edit Skor yang Berwarna Kuning)")
            
            # 2. Tabel Interaktif yang bisa diedit Manajer
            edited_df = st.data_editor(df, num_rows="dynamic")
            
            # 3. Tombol Simpan
            if st.button("Simpan Hasil Validasi"):
                st.success("Data berhasil disimpan secara permanen!")
