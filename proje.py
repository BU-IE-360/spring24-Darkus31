import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import datetime

# Dosya yolu
file_path = 'c:/Users/bymar/Desktop/IE360/production.xlsx'
data = pd.read_excel(file_path)

# Tarih ve saat bilgilerini datetime objesine dönüştürme
data['datetime'] = pd.to_datetime(data['date']) + pd.to_timedelta(data['hour'], unit='h')

# Saat ve haftanın günü gibi yeni özellikler türetme
data['hour_of_day'] = data['datetime'].dt.hour
data['day_of_week'] = data['datetime'].dt.weekday

# Sinüs ve kosinüs dönüşümleri ekleme
data['sin_hour'] = np.sin(2 * np.pi * data['hour_of_day']/24)
data['cos_hour'] = np.cos(2 * np.pi * data['hour_of_day']/24)

# Özellikler ve hedef değişkeni ayırma
X = data[['sin_hour', 'cos_hour', 'day_of_week']]  # Sinüs ve kosinüs özelliklerini de kullanma
y = data['production']

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lineer regresyon modelini başlatma ve eğitme
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Test setinde tahmin yapma
linear_y_pred = linear_model.predict(X_test)

# Lineer modeli değerlendirme
linear_mse = mean_squared_error(y_test, linear_y_pred)
linear_r2 = r2_score(y_test, linear_y_pred)

# Rastgele Orman modelini başlatma ve eğitme
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Test setinde tahmin yapma
rf_y_pred = random_forest_model.predict(X_test)

# Rastgele Orman modelini değerlendirme
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)

# Değerlendirme metriklerini yazdırma
print("Linear Regression Mean Squared Error:", linear_mse)
print("Linear Regression R-squared:", linear_r2)
print("Random Forest Mean Squared Error:", rf_mse)
print("Random Forest R-squared:", rf_r2)
