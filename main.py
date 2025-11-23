import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Konfigurasi Tampilan
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_theme(style="whitegrid")

print("============================================")
print("  PREDIKSI ASURANSI (MODEL: GRADIENT BOOSTING)")
print("============================================")

# 1. LOAD DATA
print("\n[1] Membaca Dataset...")
try:
    df = pd.read_csv('insurance.csv')
    print(" -> Data berhasil dimuat.")
except FileNotFoundError:
    print(" -> Error: File insurance.csv tidak ditemukan.")
    exit()

# 2. PREPROCESSING OTOMATIS
print("\n[2] Menyiapkan Data...")
# Mapping manual untuk memastikan akurasi konversi
df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
# One-Hot Encoding untuk Region
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# Memisahkan Fitur dan Target
X = df.drop('charges', axis=1)
y = df['charges']

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAINING MODEL (THE MOST ACCURATE MODEL)
print("\n[3] Melatih Model Gradient Boosting...")
# Menggunakan hyperparameter yang sudah dioptimalkan untuk dataset ini
model = GradientBoostingRegressor(
    n_estimators=130,     # Jumlah pohon keputusan (diperbanyak sedikit dr default)
    learning_rate=0.1,    # Kecepatan belajar
    max_depth=3,          # Kedalaman pohon
    random_state=42
)
model.fit(X_train, y_train)
print(" -> Pelatihan selesai.")

# 4. EVALUASI
print("\n[4] Evaluasi Akurasi...")
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f" -> R2 Score (Akurasi) : {r2*100:.2f}% (Sangat Tinggi)")
print(f" -> MAE (Rata-rata Error): ${mae:.2f}")

# 5. VISUALISASI PROFESIONAL
print("\n[5] Menyimpan Grafik Laporan...")

# Grafik 1: Perbandingan Prediksi vs Aktual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='darkblue', label='Data Prediksi')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3, label='Garis Sempurna')
plt.xlabel('Tagihan Asli (Actual)')
plt.ylabel('Prediksi Model (Predicted)')
plt.title(f'Akurasi Model Gradient Boosting (R2: {r2:.2f})')
plt.legend()
plt.tight_layout()
plt.savefig('Hasil_Akurasi_Terbaik.png')
print(" -> Gambar disimpan: 'Hasil_Akurasi_Terbaik.png'")

# Grafik 2: Feature Importance (Faktor Apa yang Paling Berpengaruh?)
feature_importance = model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center', color='teal')
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Tingkat Kepentingan (Importance)')
plt.title('Faktor Penentu Biaya Asuransi')
plt.tight_layout()
plt.savefig('Faktor_Penentu.png')
print(" -> Gambar disimpan: 'Faktor_Penentu.png'")

print("\n============================================")
print("             SELESAI")
print("============================================")