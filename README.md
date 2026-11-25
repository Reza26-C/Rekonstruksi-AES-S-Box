# Rekonstruksi AES S-Box — Chaotic Map + Gradient Boosting

Aplikasi Streamlit untuk rekonstruksi AES S-Box non-polinomial menggunakan 2D Logistic-Sine Chaotic Map, `GradientBoostingRegressor`, dan optimasi permutasi swap, dilengkapi uji metrik kriptografi, demo enkripsi/dekripsi AES-128 (teks & gambar), dan ekspor Excel/CSV.

## Menjalankan Aplikasi

Klik dua kali `run_app.bat` atau jalankan di PowerShell:

```powershell
cd "d:\Projek Kryptografi"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Menjalankan Pengujian

```powershell
cd "d:\Projek Kryptografi"
.\.venv\Scripts\python.exe tests\test_all.py
```
