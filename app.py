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

# Model yang digunakan (Gemini 3.5 Flash - GA)
MODEL_ID = "gemini-3.5-flash"

# Parameter untuk memaksimalkan KONSISTENSI hasil antar-run.
# CATATAN PENTING: temperature=0 + top_k=1 TIDAK direkomendasikan untuk
# model Gemini 3.x (termasuk gemini-3.5-flash) karena model ini punya mode
# "thinking" internal — kombinasi itu bisa membuat model terjebak di proses
# berpikir tanpa pernah mengeluarkan jawaban akhir (response.text jadi
# kosong). Gunakan thinking_level + seed sebagai gantinya.
GENERATION_CONFIG_TEXT = types.GenerateContentConfig(
    seed=42,
    thinking_config=types.ThinkingConfig(thinking_level="low"),
)

GENERATION_CONFIG_JSON = types.GenerateContentConfig(
    seed=42,
    thinking_config=types.ThinkingConfig(thinking_level="low"),
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
          f"✅ **{deskripsi_agen}:** Selesai! Pendinginan 12 detik"
          " (Anti-Limit)..."
      )
      time.sleep(12)
      return teks_hasil
    except Exception as e:
      pesan_error = str(e).upper()
      if any(
          k in pesan_error
          for k in ["503", "429", "RESOURCE_EXHAUSTED", "UNAVAILABLE"]
      ):
        log_ui.write(
            f"⚠️ **{deskripsi_agen}:** Peladen sibuk. Menunggu 20 detik..."
        )
        time.sleep(20)
      else:
        if percobaan == maksimal_percobaan - 1:
          raise
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
            " dokumen, tanpa asumsi atau tambahan opini.\n\n"
            "Ekstrak dari PDF secara sistematis dan lengkap:\n"
            "1. Masalah Utama (kondisi awal, data pendukung)\n"
            "2. Solusi/Perbaikan yang dilakukan\n"
            "3. Bukti Visual yang tersedia (foto before/after, diagram,"
            " grafik — sebutkan ada/tidaknya masing-masing)\n"
            "4. Hasil Angka Nyata (Saving) — sebutkan satuan dan"
            " periode pengukuran bila ada\n\n"
            "Jika suatu elemen tidak ditemukan di dokumen, nyatakan dengan"
            " jelas 'TIDAK DITEMUKAN' — jangan mengarang."
        )
        laporan_agen_1 = panggil_ai_dengan_retry(
            [gemini_file, prompt_1], "Pengekstrak Bukti [1/5]", status_box
        )

        # --- 2. Jaksa (Kritik) ---
        prompt_2 = (
            "Kamu berperan sebagai Jaksa yang skeptis dalam audit Kaizen."
            " Tugasmu HANYA mencari kelemahan berbasis fakta yang ada,"
            " bukan mengarang tuduhan.\n\n"
            f"Fakta Kasus:\n{laporan_agen_1}\n\n"
            "Identifikasi: kelemahan bukti, celah logika antara masalah dan"
            " solusi, kurangnya data pendukung, atau potensi manipulasi"
            " angka saving. Sertakan alasan yang merujuk ke fakta di atas."
        )
        dakwaan_jaksa = panggil_ai_dengan_retry(
            prompt_2, "Jaksa Penilai [2/5]", status_box
        )

        # --- 3. Pembela ---
        prompt_3 = (
            "Kamu berperan sebagai Pengacara Pembela dalam audit Kaizen."
            " Tugasmu membela HANYA berdasarkan fakta yang tersedia, bukan"
            " asumsi baik yang tidak berdasar.\n\n"
            f"Fakta:\n{laporan_agen_1}\n\nKritik Jaksa:\n{dakwaan_jaksa}\n\n"
            "Bantah kritik yang tidak berdasar dan soroti nilai tambah yang"
            " sudah terbukti dari fakta di atas."
        )
        pembelaan_pengacara = panggil_ai_dengan_retry(
            prompt_3, "Pengacara Pembela [3/5]", status_box
        )

        # --- 4. Hakim Agung (Skoring 21 Poin) ---
        prompt_4 = f"""Kamu adalah Hakim Agung penilaian Kaizen yang wajib bersikap objektif, konsisten, dan hanya menilai berdasarkan bukti tertulis — bukan asumsi.

ATURAN PENILAIAN:
- Beri skor SESUAI pilihan yang tersedia per kriteria (jangan beri skor di luar pilihan).
- Jika bukti tidak ditemukan/lemah, beri skor terendah pada kriteria tsb.
- Justifikasi WAJIB merujuk fakta konkret dari dokumen, bukan opini umum.
- Bersikap ketat: skor tinggi hanya untuk bukti yang benar-benar kuat dan lengkap.

Fakta: {laporan_agen_1}
Kritik Jaksa: {dakwaan_jaksa}
Pembelaan: {pembelaan_pengacara}

Evaluasi 21 kriteria berikut (nilai dalam kurung adalah pilihan skor yang SAH):
1. 5G [0,1,2] | 2. Losses Measurement [0,1,2] | 3. 5W1H [0,1,2] | 4. Visualisasi [0,1,2] | 5. Target SMART [0,2] | 6. Fishbone 4M [0,1,2] | 7. Pemetaan 4M [0,1,2] | 8. Hubungan Akar Penyebab [0,1,2] | 9. Bukti Akar Penyebab [0,3,5] | 10. Ketepatan Root Cause [0,1,2] | 11. Action Plan PIC [0,1,2] | 12. Rencana Perbaikan [0,1,2] | 13. Form Usulan Perbaikan [0,3,5] | 14. Pelaksanaan Action Plan [0,1,2] | 15. Dokumentasi Pelaksanaan [0,5,8] | 16. Pencapaian Target [0,1] | 17. Pengecekan Hasil [0,3,5] | 18. Kelengkapan Standardisasi [0,3,5] | 19. Validasi Standardisasi [0,1,2] | 20. Tindak Lanjut Sosialisasi [0,3,5] | 21. Replikasi [0,3,5]

Keluarkan HANYA JSON array valid, tanpa teks lain, dengan skema persis:
[{{"no": 1, "kriteria": "5G", "skor": 2, "justifikasi": "alasan singkat merujuk fakta"}}]"""
        raw_hakim = panggil_ai_dengan_retry(
            prompt_4,
            "Hakim Agung [4/5]",
            status_box,
            config=GENERATION_CONFIG_JSON,
        )
        hasil_hakim_json = bersihkan_dan_parse_json(raw_hakim)

        # --- 5. Analis Impact ---
        prompt_5 = (
            "Kamu adalah Analis Impact yang menilai dampak operasional dari"
            " dokumen Kaizen ini secara objektif berdasarkan bukti"
            " tertulis saja.\n\n"
            "Evaluasi 8 kategori impact: Gas/Steam, Material Balance,"
            " Manpower, Downtime, Waktu Kerja, Overtime, Listrik, Air.\n"
            "Untuk setiap kategori, status HARUS salah satu dari:"
            " 'YA' (ada dampak terbukti), 'TIDAK' (tidak ada/tidak"
            " disebutkan), atau 'TIDAK RELEVAN'.\n\n"
            "Keluarkan HANYA JSON array valid dengan skema persis:\n"
            '[{"kategori": "Air", "status": "TIDAK", "keterangan":'
            ' "alasan singkat merujuk dokumen"}]'
        )
        raw_analis = panggil_ai_dengan_retry(
            [gemini_file, prompt_5],
            "Analis Kesan [5/5]",
            status_box,
            config=GENERATION_CONFIG_JSON,
        )
        hasil_saving_json = bersihkan_dan_parse_json(raw_analis)

        status_box.update(
            label="✅ Analisis Selesai!", state="complete", expanded=False
        )

        # --- PEMROSESAN DATA TABEL ---
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
            {"Peranan": "Ejen Pengekstrak", "Laporan": laporan_agen_1},
            {"Peranan": "Ejen Jaksa Penilai", "Laporan": dakwaan_jaksa},
            {
                "Peranan": "Ejen Pengacara Pembela",
                "Laporan": pembelaan_pengacara,
            },
            {"Peranan": "Ejen Hakim Agung", "Laporan": raw_hakim},
            {"Peranan": "Ejen Analis Kesan", "Laporan": raw_analis},
        ]

        st.session_state.proses_selesai = True
        st.rerun()

      except Exception as e:
        status_box.update(label="❌ Terjadi Kesalahan", state="error")
        st.error(f"**Pesan Ralat:** `{e}`")
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

  st.subheader("💰 2. Tabel Validasi Saving (8 Kategori)")
  edited_saving = st.data_editor(
      st.session_state.df_saving,
      num_rows="dynamic",
      use_container_width=True,
      key="tabel_saving",
  )

  with st.expander("📜 Lihat Transkrip Lengkap Multi-Agent"):
    for entri in st.session_state.transkrip:
      st.markdown(f"**{entri['Peranan']}**")
      st.text(entri["Laporan"])
      st.divider()

  output = io.BytesIO()
  with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
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
        label="📥 Unduh Laporan Lengkap (Excel 3 Sheet)",
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
      st.session_state.df_rubrik = pd.DataFrame()
      st.session_state.df_saving = pd.DataFrame()
      st.session_state.transkrip = []
      st.session_state.nama_file = "Dokumen_Kaizen"
      st.rerun()
