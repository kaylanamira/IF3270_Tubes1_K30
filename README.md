# Tugas Besar 1 IF3270 Pembelajaran Mesin : MiawLentera

## Deskripsi Singkat
Repositori ini berisi implementasi Feedforward Neural Network (FFNN) yang dilakukan from scratch. FFNN adalah salah satu jenis Artificial Neural Network (ANN), yaitu model machine learning yang meniru cara kerja otak manusia dalam memproses informasi dengan memanfaatkan neuron dan hubungan antarneuron. Terdapat dua variasi FFNN yang diimplementasikan, satu menggunakan autograd dan satu tidak. Model yang dirancang akan digunakan untuk memprediksi digit angka pada dataset MNIST-784.

Terdapat beberapa spesifikasi utama yang harus dipenuhi dalam pengembangan model Feedforward Neural Network, antara lain:

1. Arsitektur Jaringan : FFNN harus mendukung jumlah layer dan jumlah neuron per layer yang dapat dikonfigurasi.
2. Fungsi Aktivasi : Model harus mendukung fungsi aktivasi Linear, ReLU, Sigmoid, Tanh, dan Softmax.

3. Fungsi Loss : Model harus mendukung fungsi loss Mean Squared Error (MSE), Binary Cross-Entropy, dan Categorical Cross-Entropy.
4. Inisialisasi Bobot : Bobot dan bias dapat diinisialisasi menggunakan metode Zero Initialization, Uniform Distribution, dan Normal Distribution.
5. Forward Propagation : Implementasi perhitungan keluaran jaringan berdasarkan input yang diberikan.
6. Backward Propagation : Implementasi perhitungan gradien bobot menggunakan konsep chain rule.
Weight Update : Pembaruan bobot menggunakan gradient descent.
7. Visualisasi Model : Model harus dapat menampilkan struktur jaringan, bobot, dan distribusi gradien dalam bentuk graf.
8. Penyimpanan dan Pemuatan Model : Model dapat disimpan dan dimuat kembali untuk penggunaan selanjutnya.

## Cara Setup dan Run Program

Berikut merupakan langkah-langkah untuk melatih model dan menggunakan model dalam prediksi:

### 1. Clone repositori ini
`git clone https://github.com/kaylanamira/IF3270_Tubes1_K30.git`

### 2. Buat notebook baru dan import dependensi berikut:
```
import importlib
import miaw
importlib.reload(miaw)
from miaw import FFNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score
```

### 3. Fetch dataset MNIST-784 dan split dataset menjadi training set dan validation set
```
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

encoder = OneHotEncoder(sparse_output=False, categories='auto')
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)
```

### 4. Fetch test dataset
```
# Random test sample from Kaggle
test_url = 'https://drive.google.com/uc?id=1VwGTcs1guwJTnUHKdOkMgDUtuW18DbEp'
test_df = pd.read_csv(test_url)

y_test = test_df["label"].values 
X_test = test_df.drop(columns=["label"]).values
print(X_test.shape, y_test.shape)
```

### 5. Bruat arsitektur model dan latih model
```
# Membuat model dengan 1 hidden layer, fungsi aktivasi relu dan softmax, dll.
d1_model = FFNN([784, 64, 10], ['relu', 'softmax'],loss_function='cce', use_rmsnorm=False, reg_type="l2", weight_init='he')
d1_history = d1_model.fit(X_train, y_train, X_val, y_val, epochs=50, batch_size=64, lr=0.05, verbose=0)
```

### 6. Lakukan prediksi dan lihat akurasi model pada validation set dan test set
```
d1_test_pred = d1_model.predict(X_test)
d1_val_pred = d1_model.predict(X_val)

y_val_decoded = np.argmax(y_val, axis=1)

val_accuracy = accuracy_score(d1_val_pred, y_val_decoded)
test_accuracy = accuracy_score(d1_test_pred, y_test)
print(f"Val accuracy: {val_accuracy * 100:.2f}%, Test accuracy: {test_accuracy * 100:.2f}%")

d1_model.plot_weight_distribution([0, 1])
d1_model.plot_loss_history(d1_history)
```
Jika semua berjalan dengan benar, seharusnya akan dihasilkan output seperti:

`Val accuracy: 93.28%, Test accuracy: 95.28%`

## Pembagian Tugas

### Muhammad Neo Cicero Koda (13522108)
- Forward propagation (autograd)
- Backward propagation (autograd)
- Activation functions
- Weight initialization
- Laporan dan testing

### Angelica Kierra Ninta Gurning (13522048)
- FFNN Model (non-autograd)
- Regularization
- Normalizaton
- Laporan

### Kayla Namira Mariadi
- FFNN Model Visualization
- FFNN Model (non-autograd)
- Laporan
- Testing
