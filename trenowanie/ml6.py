import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from PIL import Image
import joblib
import os

# 1. Załadowanie danych do treniengu
train_df = pd.read_csv('D:/Marcin/ML/A_Z Handwritten Data.csv', header=None)

# Pierwsza kolumna to etykieta (class_label), reszta to piksele
# każdy wiersz zawiera 785 znaków oddzielonych przecinkami, 1 jest kluczem(etykietą) reszta to wartości pikseli od 0 do 225

X = train_df.iloc[:, 1:].values  # Dane wejściowe (piksele)
y = train_df.iloc[:, 0].values   # Etykiety klas (liczby)

# 2. Załadowanie mappings.txt i utworzenie mapy etykiet klas na litery
mappings = pd.read_csv('D:/Marcin/ML/mapping.txt', header=None, sep=' ', names=['key', 'value'])

# Tworzenie słownika {key: value}, gdzie key to klasa, a value to litera
mappings_dict = mappings.set_index('key').to_dict()['value']

# Zmapowanie klas numerycznych na litery
y = np.array([mappings_dict[label] for label in y])

# 3. Podział na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# 4. Budowa i trenowanie modelu
model = RandomForestClassifier(n_estimators=100, random_state=42)
# model 4 był wytrenowany na n = 100, ze stanem losowym 42
model.fit(X_train, y_train)

# 5. Ewaluacja modelu
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6. Zapisanie model do pliku pkl
try:
    joblib.dump(model, 'D:/Marcin/ML/modele/ml_5.pkl')
    print("Model został zapisany pomyślnie!")
except Exception as e:
    print(f"Wystąpił błąd podczas zapisywania modelu: {e}")
