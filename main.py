import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("============================================")
print("   PROGRAM PREDIKSI BIAYA ASURANSI")
print("============================================")

print("\n[1] Membaca File Data...")
try:
    df = pd.read_csv('insurance.csv')
    print(" -> Berhasil! File 'insurance.csv' ditemukan.")
except FileNotFoundError:
    print(" -> Error: File tidak ketemu.")
    exit()

print("\n[2] Mengecek Informasi Data Secara Rinci...")
print(f" -> Ukuran Data: {df.shape[0]} Baris dan {df.shape[1]} Kolom.")
print(f" -> Data Kosong: {df.isnull().sum().sum()}")

print("\n--- Statistik Dasar ---")
print(df.describe())

print("\n--- Detail Jumlah Data Kategori ---")
print("1. Jenis Kelamin:\n", df['sex'].value_counts())
print("\n2. Perokok:\n", df['smoker'].value_counts())
print("\n3. Wilayah:\n", df['region'].value_counts())

print("\n\n[3] Mengubah Data Huruf Menjadi Angka...")

df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df = pd.get_dummies(df, columns=['region'], drop_first=True)

print(" -> Data berhasil diubah! Contoh 5 baris teratas:")
print(df.head())

print("\n\n[4] Mulai Melatih Model AI...")

X = df.drop('charges', axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print(" -> AI selesai belajar.")

print("\n[5] Menguji Kecerdasan AI...")

y_pred = model.predict(X_test)
skor_akurasi = r2_score(y_test, y_pred)

print(f" -> Akurasi Model (R2 Score): {skor_akurasi:.4f}")

print("\n[6] Membuat Grafik Laporan...")

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['smoker'], y=df['charges'])
plt.title('Analisis: Apakah Merokok Mempengaruhi Biaya?')
plt.xlabel('Status Perokok (0=Tidak, 1=Ya)')
plt.ylabel('Biaya Asuransi')
plt.savefig('Laporan_1_Efek_Rokok.png')
print(" -> Gambar 1 disimpan: 'Laporan_1_Efek_Rokok.png'")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Evaluasi: Prediksi AI vs Data Asli')
plt.xlabel('Biaya Asli')
plt.ylabel('Prediksi AI')
plt.savefig('Laporan_2_Akurasi_Model.png')
print(" -> Gambar 2 disimpan: 'Laporan_2_Akurasi_Model.png'")

print("\n============================================")
print("       PROGRAM SELESAI DENGAN SUKSES")
print("============================================")
