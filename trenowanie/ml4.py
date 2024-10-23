import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from PIL import Image
import joblib
import os

# 1. Załadowanie danych
train_df = pd.read_csv('D:/Marcin/ML/A_Z Handwritten Data.csv', header=None)

# Pierwsza kolumna to etykieta (class_label), reszta to piksele
X = train_df.iloc[:, 1:].values  # Dane wejściowe (piksele)
y = train_df.iloc[:, 0].values   # Etykiety klas (liczby)

# 2. Załadowanie mappings.txt i utworzenie mapy etykiet klas na litery
mappings = pd.read_csv('D:/Marcin/ML/mappings.txt', header=None, sep=' ', names=['key', 'value'])

# Tworzenie słownika {key: value}, gdzie key to klasa, a value to litera
mappings_dict = mappings.set_index('key').to_dict()['value']

# Zmapowanie klas numerycznych na litery
y = np.array([mappings_dict[label] for label in y])

# 3. Podział na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Budowa i trenowanie modelu
model = RandomForestClassifier(n_estimators=10, random_state=2)
model.fit(X_train, y_train)

# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
# git to wyzej


# 5. Ewaluacja modelu
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Zapisz model
#joblib.dump(model, 'letter_recognition_model.pkl')
try:
    joblib.dump(model, 'D:/Marcin/ML/letter_recognition_model.pkl')
    print("Model został zapisany pomyślnie!")
except Exception as e:
    print(f"Wystąpił błąd podczas zapisywania modelu: {e}")





# 6. Funkcja do wczytania obrazka i przewidzenia litery
def predict_letter(image_path):
    # Wczytanie obrazka 28x28 px
    img = Image.open(image_path).convert('L')  # Konwersja na obraz czarno-biały
    img = img.resize((28, 28))  # Zmiana rozmiaru na 28x28 pikseli
    img_data = np.array(img).reshape(1, -1)  # Zmiana rozmiaru na wektor 1x784

    # Normalizacja wartości pikseli do zakresu 0-255
    img_data = img_data / 255.0

    # Wczytanie modelu
    model = joblib.load('D:/Marcin/ML/letter_recognition_model.pkl')

    # Predykcja litery
    predicted_letter = model.predict(img_data)
    return predicted_letter[0]







# Przykład: Testowanie funkcji na wybranym obrazku
image_path = 'D:/Marcin/ML/a1.png'  # Ścieżka do obrazka 28x28 px
predicted_letter = predict_letter(image_path)
print(f"Przewidziana litera: {predicted_letter}")

# Wyświetlenie testowego obrazka (opcjonalne)
img = Image.open(image_path)
import matplotlib.pyplot as plt
plt.imshow(img, cmap='gray')
plt.title(f"Przewidziana litera: {predicted_letter}")
plt.show()
