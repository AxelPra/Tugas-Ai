import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Biar tampilan tabel di terminal lebar dan enak dibaca
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("============================================")
print("   PROGRAM PREDIKSI BIAYA ASURANSI")
print("============================================")

# --- 1. MEMUAT DATA (DATA MINING) ---
print("\n[1] Membaca File Data...")
try:
    df = pd.read_csv('insurance.csv')
    print(" -> Berhasil! File 'insurance.csv' ditemukan.")
except FileNotFoundError:
    print(" -> Error: File tidak ketemu. Pastikan ada di folder yang sama.")
    exit()

# --- 2. CEK DETAIL DATA (DATA EXPLORATION) ---
print("\n[2] Mengecek Informasi Data Secara Rinci...")

# Cek dimensi data
jml_baris, jml_kolom = df.shape
print(f" -> Ukuran Data: {jml_baris} Baris dan {jml_kolom} Kolom.")

# Cek apakah ada data kosong
jumlah_kosong = df.isnull().sum().sum()
print(f" -> Data Kosong (Missing Values): {jumlah_kosong}")

# Cek statistik sederhana (Rata-rata, Min, Max)
print("\n--- Statistik Dasar (Angka) ---")
print(df.describe())

# Cek sebaran data kategori (Contoh: Berapa perokok vs bukan)
print("\n--- Detail Jumlah Data Kategori ---")
print("1. Jenis Kelamin:\n", df['sex'].value_counts())
print("\n2. Perokok:\n", df['smoker'].value_counts())
print("\n3. Wilayah:\n", df['region'].value_counts())

# --- 3. PERSIAPAN DATA (FEATURE ENGINEERING) ---
print("\n\n[3] Mengubah Data Huruf Menjadi Angka...")

# Kita ubah kolom 'sex' dan 'smoker' jadi angka 0 dan 1
# sex: female jadi 0, male jadi 1
df['sex'] = df['sex'].map({'female': 0, 'male': 1})

# smoker: no jadi 0, yes jadi 1 (Ini faktor terpenting!)
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})

# region: Ubah jadi banyak kolom (One Hot Encoding) biar model paham
df = pd.get_dummies(df, columns=['region'], drop_first=True)

print(" -> Data berhasil diubah! Contoh 5 baris teratas:")
print(df.head())

# --- 4. MELATIH AI (MODELING) ---
print("\n\n[4] Mulai Melatih Model AI...")

# X = Data pendukung (Umur, BMI, Rokok, dll)
X = df.drop('charges', axis=1)
# y = Target yang mau ditebak (Biaya Asuransi)
y = df['charges']

# Bagi data: 80% buat Latihan, 20% buat Ujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Panggil otak AI-nya (Linear Regression)
model = LinearRegression()
# Suruh AI belajar dari data latihan
model.fit(X_train, y_train)

print(" -> AI selesai belajar.")

# --- 5. EVALUASI HASIL (EVALUATION) ---
print("\n[5] Menguji Kecerdasan AI...")

# Minta AI menebak harga dari data ujian
y_pred = model.predict(X_test)

# Hitung skor akurasinya (R-Squared)
# Nilai 1.0 = Sempurna, 0.0 = Jelek
skor_akurasi = r2_score(y_test, y_pred)

print(f" -> Akurasi Model (R2 Score): {skor_akurasi:.4f}")
print("    (Nilai di atas 0.70 artinya model sangat bagus untuk pemula)")

# --- 6. BUAT LAPORAN GAMBAR (VISUALIZATION) ---
print("\n[6] Membuat Grafik Laporan...")

# Grafik 1: Perbandingan Biaya Perokok
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['smoker'], y=df['charges'])
plt.title('Analisis: Apakah Merokok Mempengaruhi Biaya?')
plt.xlabel('Status Perokok (0=Tidak, 1=Ya)')
plt.ylabel('Biaya Asuransi')
plt.savefig('Laporan_1_Efek_Rokok.png')
print(" -> Gambar 1 disimpan: 'Laporan_1_Efek_Rokok.png'")

# Grafik 2: Hasil Prediksi vs Asli
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.6)
# Bikin garis diagonal merah (Garis Ideal)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Evaluasi: Prediksi AI vs Data Asli')
plt.xlabel('Biaya Asli')
plt.ylabel('Prediksi AI')
plt.savefig('Laporan_2_Akurasi_Model.png')
print(" -> Gambar 2 disimpan: 'Laporan_2_Akurasi_Model.png'")

print("\n============================================")
print("       PROGRAM SELESAI DENGAN SUKSES")
print("============================================")