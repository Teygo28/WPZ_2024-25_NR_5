import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from PIL import Image
import joblib
import matplotlib.pyplot as plt

# 1. Funkcja do wczytania obrazka i przewidzenia litery
def predict_letter(image_path):
    
    # Wczytanie obrazka 28x28 px i konwersja na obraz czarno-biały. Zmiana rozmiaru na 28x28 pikseli. Zmiana rozmiaru na wektor 1x784
    img = Image.open(image_path).convert('L') 
    img = img.resize((28, 28))  
    img_data = np.array(img).reshape(1, -1)  

    # Normalizacja wartości pikseli do zakresu 0-255 - rozpozanwanie kolorowych obrazów
    img_data = [[255 if x > 127 else 0 for x in sublist] for sublist in img_data]
  
    # Wypisanie danych do testów
    # print(img_data)
    # Wczytanie modelu
    model = joblib.load('D:/Marcin/ML/modele/ml_4.pkl')

    # Predykcja litery
    predicted_letter = model.predict(img_data)
    return predicted_letter[0]

# Przykład: Testowanie funkcji na wybranym obrazku. Ścieżka do obrazka 28x28 px

image_path = 'D:/Marcin/ML/img/jj.png'              #    <-------------  tu zdjęcie

predicted_letter = predict_letter(image_path)
print(f"Przewidziana litera: {predicted_letter}")

#  Stworzenie wykresu do prezentacji graficznej o wymiarach 1 wiersz, 2 kolumny
img = Image.open(image_path)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Pierwszy subplot - obraz w skali szarości
axes[0].imshow(img.convert("L"), cmap='gray')
axes[0].set_title("Skala szarości")

axes[1].imshow(img, cmap='gray')
plt.title(f"Przewidziana litera: {predicted_letter}")
plt.show()

