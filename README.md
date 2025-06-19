# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Business Understanding

**Jaya Jaya Institut** merupakan salah satu institusi pendidikan perguruan tinggi yang telah berdiri sejak tahun 2000. Hingga saat ini, institut ini telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat cukup banyak siswa yang tidak menyelesaikan pendidikannya alias **dropout**.

Tingginya jumlah dropout menjadi masalah serius bagi institusi pendidikan karena dapat memengaruhi reputasi, efisiensi operasional, serta keberhasilan proses belajar-mengajar. Oleh karena itu, pihak Jaya Jaya Institut ingin mendeteksi **lebih awal** siswa yang berpotensi dropout agar dapat diberikan **bimbingan khusus**.

Sebagai calon data scientist masa depan, kita diminta untuk membantu Jaya Jaya Institut menyelesaikan masalah ini dengan menggunakan pendekatan **machine learning dan dashboard analitik**.

### Permasalahan Bisnis

Jaya Jaya Institut menghadapi tantangan serius dalam hal tingginya angka mahasiswa dropout yang berpotensi mengganggu reputasi akademik, akreditasi, serta efisiensi proses pembelajaran. Institusi kesulitan dalam:
- Mengidentifikasi faktor utama yang mendorong mahasiswa untuk berhenti studi di tengah jalan.
- Tidak adanya sistem prediksi berbasis data yang dapat memberikan peringatan dini bagi mahasiswa yang berisiko tinggi dropout.
- Kesulitan dalam monitoring performa akademik mahasiswa secara real-time dan menyeluruh dari sisi akademik dan sosial.
Oleh karena itu, institusi memerlukan pendekatan analitik dan sistem machine learning untuk membantu pengambilan keputusan yang berbasis data.

### Cakupan Proyek

- Mengidentifikasi faktor-faktor yang memengaruhi status mahasiswa (Graduate, Dropout, Enrolled)
- Membangun model prediksi status mahasiswa menggunakan supervised machine learning
- Menyusun dashboard analitik menggunakan Looker Studio untuk memvisualisasikan insight penting dari data
- Memberikan rekomendasi berbasis data untuk pengambilan keputusan akademik

---

## Persiapan

**Sumber data:**  
- Dataset asli: 
https://drive.google.com/file/d/1fXxRxV11Oa7lqhWHRralQx5iXMNyDk5N/view?usp=sharing
- Dataset bersih dan delimiter diperbaiki: 
https://drive.google.com/file/d/1kDE2zUzRWjqnP5pRK4l3yOJI_cTN-_Ua/view?usp=sharing

**Setup environment:**  
Library utama: `pandas`, `numpy`, `scikit-learn`, `streamlit`

### Membuat dan Mengaktifkan Virtual Environment (venv)
Buka terminal

# Membuat virtual environment
python -m venv venv

# Mengaktifkan (Windows)
venv\Scripts\activate

# Mengaktifkan (Mac/Linux)
source venv/bin/activate

# Menginstal Dependensi dari requirements.txt
pip install -r requirements.txt

## Data Understanding
Mengenali isi deskripsi data, nilai unik status, jumlah data missing, dan tahap EDA serta visualisasi data.

**Insight:**  
- Jumlah baris: 4425 baris, dengan baris awal nama kolom dan jumlah baris dataaa 4424
- Dataset terdiri dari informasi akademik dan sosial mahasiswa, seperti nilai masuk, beasiswa, status pembayaran, serta kinerja per semester.
- Walaupun tidak ada NaN tetap gunakan kode hapus untuk kerapian.
- Target variabel (Status) menunjukkan distribusi yang tidak seimbang: lebih banyak yang graduate, dengan dropout nilai di tengah dan paling rendah nilai enrolled.
- Fitur Curricular_units_1st_sem_approved,Curricular_units_1st_sem_grade, Curricular_units_1st_sem_approved
Curricular_units_2nd_sem_approved,Curricular_units_2nd_sem_grade, Curricular_units_2nd_sem_approved, dan Tuition_fees_up_to_date  terlihat berpengaruh distribusinya terhadap status.
- Kelas Graduate mendominasi, artinya banyak mahasiswa berhasil menyelesaikan studi. Kelas nilai di tengah dropout, dan enrolled data minoritas


## Data Preparation
- Memetakan label `status` menjadi `'dropout: 0'`, `'enrolled: 1'`, dan `'graduate:2'`
- Dilakukan encoding kolom kategorikal dan normalisasi fitur numerik saat proses modeling di notebook.

**Insight:**  
- Fitur-fitur kategorikal yang berpengaruh seperti Status, Scholarship_holder, Debtor, dll. telah dikonversi menjadi bentuk numerik agar bisa diproses model.
- Fitur numerik seperti Admission_grade, Age_at_enrollment, dan Semester Grades bisa memiliki skala yang berbeda. Standardisasi membantu model belajar secara seimbang dan mempercepat konvergensi.
- Data dibagi menjadi 80% untuk pelatihan dan 20% untuk pengujian, dengan random seed yang dikunci (random_state=42).

## Modeling

Model yang digunakan:
- Logistic Regression
- Random Forest
- KNN
- SVM
- Gradient Boosting
- AdaBoost
- Extra Trees


**Insight:**  
- 7 model diuji: Logistic Regression, Random Forest, KNN, SVM, Gradient Boosting, AdaBoost, dan Extra Trees
- Model Logistic Regression dan Gradient Boosting menunjukkan akurasi terbaik (~86–88%).
- Overall model dapat membedakan dengan cukup baik antara karyawan yang akan Resign dan yang akan Stay dengan tidak ada akurasi dibawah 80%.

## Evaluation

Dilakukan evaluasi dengan:
- Confusion matrix
- Classification report
- Feature importance
- Precision-Recall curve

**Insight:**  
- Dari semua model, model paling akurat ialah Extra Trees dengan nilai 76.84% yang bisa dibulatkan menjadi 77%
- Feature Importance mengindikasikan bahwa fitur seperti Curricular_units_1st_sem_approved,Curricular_units_1st_sem_grade, Curricular_units_1st_sem_approved
Curricular_units_2nd_sem_approved,Curricular_units_2nd_sem_grade, Curricular_units_2nd_sem_approved, dan Tuition_fees_up_to_date memiliki bobot yang sangat berpengaruh. 
- Precision-Recall Curve menunjukkan bahwa model sudah cukup untuk menangani ketidakseimbangan data dan memberikan prediksi terhadap kelas minoritas (enrolled).

## Business Dashboard

Dashboard dibuat menggunakan **Looker Studio** dengan sumber dari CSV. Visualisasi mencakup:
- Proporsi Dropout vs Enrolled vs Graduated 
- Distribusi dan Korelasi fitur seperti `Curricular_units_1st_sem_approved,Curricular_units_1st_sem_grade`, `Curricular_units_1st_sem_approved`
`Curricular_units_2nd_sem_approved`, `Curricular_units_2nd_sem_grade`, `Curricular_units_2nd_sem_approved`, `Curricular_units_2nd_sem_grade`, `Curricular_units_2nd_sem_approved`, dan `Tuition_fees_up_to_date` terhadap status

https://lookerstudio.google.com/reporting/fd80c313-6bea-4325-8f22-1a5209129af1

## Menjalankan Sistem Machine Learning
Jelaskan cara menjalankan protoype sistem machine learning yang telah dibuat. Selain itu, sertakan juga link untuk mengakses prototype tersebut.
### Menjalankan aplikasi streamlit secara lokal
- Buka terminal
- run code : streamlit run streamlit_status_prediksi.py

Jika ingin membuka tidak secara lokal, dapat diakses melalui link dibawah ini:
https://prediksi-do.streamlit.app/

## Conclusion

    - Mahasiswa yang berpotensi dropout umumnya memiliki ciri Nilai rendah atau tidak ada nilai sama sekali (0) pada fitur semester seperti:
        - Curricular_units_1st_sem_approved
        - Curricular_units_1st_sem_grade
        - Curricular_units_2nd_sem_approved
        - Curricular_units_2nd_sem_grade
        - Tidak melakukan pembayaran biaya kuliah tepat waktu (Tuition_fees_up_to_date = 0).

    - Hal ini menunjukkan bahwa faktor akademik (nilai & evaluasi) dan sosial-ekonomi (status pembayaran) menjadi penentu dominan dalam risiko dropout.
    - Model prediktif Extra Trees mampu mengklasifikasikan status mahasiswa dengan akurasi yang cukup baik dan menunjukkan fitur-fitur tersebut sebagai faktor penting.
    - Sistem prediksi ini dapat digunakan sebagai alat bantu pengambilan keputusan akademik, misalnya:
        - Memberikan intervensi awal pada mahasiswa dengan performa akademik rendah.
        - Memberikan perhatian khusus kepada mahasiswa yang memiliki kendala dalam pembayaran.
Dengan implementasi sistem ini, Jaya Jaya Institut dapat menurunkan angka dropout secara signifikan serta menjaga reputasi institusi di mata publik dan regulator pendidikan.


### Rekomendasi Action Items (Optional)

1. Dikarenakan nilai `Curricular_units_1st_sem_approved,Curricular_units_1st_sem_grade`, `Curricular_units_1st_sem_approved`
`Curricular_units_2nd_sem_approved`, `Curricular_units_2nd_sem_grade`, `Curricular_units_2nd_sem_approved`, `Curricular_units_2nd_sem_grade`, dan `Curricular_units_2nd_sem_approved`, maka hal ini berarti nilai per semester murid ada yang rendah/nihil yang mempengaruhi nilai DO dari murid. Oleh karena itu sebaiknya Memberikan intervensi pada murid yang mendapatkan nilai rendah dalam semester mereka, bisa dengan menawarkan program kelas tambahan untuk murid yang nilai rendah atau kelas susulan jika murid nilainya kosong.
2. Dikarenakan nilai `Tuition_fees_up_to_date`, hal itu mempengaruhi murid dikarenakan tidak sempat membayar biaya pendidikan tepat waktu yang mengakibatkan murid terkena DO. Oleh karena itu sebaiknya untuk mecegah DO karena telat membayar biaya, harus dicari tahu alasan belum membayar biaya dan memberikan keringanan secara case-by-case yang masuk akal untuk murid.

