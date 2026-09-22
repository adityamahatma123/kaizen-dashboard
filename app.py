import streamlit as st
import pandas as pd

# Mengatur tata letak halaman web
st.set_page_config(page_title="Dashboard Kaizen", layout="wide")

st.title("📊 Dashboard Validasi Kaizen - Manager")
st.write("Unggah dokumen Kaizen, biarkan AI menilai, dan unduh hasil validasinya.")

# 1. Fitur Upload PDF
uploaded_file = st.file_uploader("Pilih file PDF Kaizen", type="pdf")

if uploaded_file is not None:
    st.success(f"File '{uploaded_file.name}' berhasil diunggah!")
    
    if st.button("Mulai Penilaian AI"):
        with st.spinner("AI sedang memproses dokumen... (Simulasi)"):
            
            # (Nantinya kode AI Python dari Colab diletakkan di sini)
            
            # Data simulasi sementara untuk menguji antarmuka
            data_dummy = {
                "No": [1, 2, 19, 21],
                "Kriteria": ["5G", "Losses", "Validasi Standardisasi", "Replikasi"],
                "Skor AI": [2, 1, 0, 0],
                "Status": ["OTOMATIS AI", "OTOMATIS AI", "⚠️ VALIDASI MANUAL", "⚠️ VALIDASI MANUAL"],
                "Justifikasi": ["Bukti foto ada.", "Target jelas.", "Belum ada TTD Sec Head.", "Tidak ada replikasi."]
            }
            df = pd.DataFrame(data_dummy)
            
            st.write("### 📝 Hasil Penilaian (Silakan Edit Skor jika diperlukan)")
            
            # 2. Tabel Interaktif yang bisa diedit Manajer
            edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
            
            # 3. Fitur Unduh ke CSV
            csv = edited_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Unduh Hasil Validasi (CSV)",
                data=csv,
                file_name=f"Hasil_Kaizen_{uploaded_file.name}.csv",
                mime="text/csv",
            )
