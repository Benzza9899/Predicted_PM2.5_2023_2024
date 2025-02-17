import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# โหลดข้อมูล
file_path = r"F:\PM2.5 DATASET\DATAPM2.5.csv"
df = pd.read_csv(file_path)

# ตรวจสอบคอลัมน์ที่จำเป็น
required_columns = ['Day', 'Year', 'Month_x', 'PM2.5', 'TempAVG']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"ไฟล์ต้องมีคอลัมน์ {required_columns}")

# แปลง 'Day' เป็น datetime และ 'PM2.5' เป็นตัวเลข
df['Day'] = pd.to_datetime(df['Day'], format="%d/%m/%Y")
df['PM2.5'] = pd.to_numeric(df['PM2.5'], errors='coerce')
df['TempAVG'] = pd.to_numeric(df['TempAVG'], errors='coerce')

# ลบค่า NaN
df = df.dropna(subset=['Year', 'Month_x', 'PM2.5', 'TempAVG'])

# เพิ่มฟีเจอร์ใหม่
df['Day_of_Month'] = df['Day'].dt.day

df['Prev_PM2.5'] = df.groupby(['Year', 'Month_x'])['PM2.5'].shift(1)
df['Prev_PM2.5'].fillna(df['PM2.5'].mean(), inplace=True)

# เลือกข้อมูลปี 2020-2024
df = df[(df['Year'] >= 2020) & (df['Year'] <= 2024)]

# ตรวจสอบว่ามีข้อมูลปี 2023-2024 หรือไม่
print("Year distribution:")
print(df['Year'].value_counts())

# แยกข้อมูล Train (2020-2022) และ Test (2023-2024)
df_train = df[df['Year'] <= 2022]
df_test = df[df['Year'] >= 2023]

# ตรวจสอบว่าข้อมูลถูกต้อง
if df_test.empty:
    raise ValueError("ไม่มีข้อมูลสำหรับปี 2023-2024 หลังจากการกรอง")

# ปรับค่า Feature Scaling
scaler = StandardScaler()
df_train.loc[:, ['Year', 'Month_x', 'TempAVG', 'Day_of_Month', 'Prev_PM2.5']] = scaler.fit_transform(df_train[['Year', 'Month_x', 'TempAVG', 'Day_of_Month', 'Prev_PM2.5']])
df_test.loc[:, ['Year', 'Month_x', 'TempAVG', 'Day_of_Month', 'Prev_PM2.5']] = scaler.transform(df_test[['Year', 'Month_x', 'TempAVG', 'Day_of_Month', 'Prev_PM2.5']])

X_train, y_train = df_train[['Year', 'Month_x', 'TempAVG', 'Day_of_Month', 'Prev_PM2.5']].values, df_train['PM2.5'].values
X_test, y_test = df_test[['Year', 'Month_x', 'TempAVG', 'Day_of_Month', 'Prev_PM2.5']].values, df_test['PM2.5'].values

# ใช้ Decision Tree พร้อม Hyperparameter tuning
model = DecisionTreeRegressor(random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ประเมินผล
test_r2 = r2_score(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)
print(f"Decision Tree - R²: {test_r2:.6f}, MSE: {test_mse:.6f}")

# บันทึกผลลัพธ์
df_test['Predicted_PM2.5'] = y_pred
df_test[['Day', 'Year', 'PM2.5', 'Predicted_PM2.5']].to_csv("Predicted_PM2.5_2023_2024.csv", index=False)

# สร้างกราฟ
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_test, x='Day', y='PM2.5', label='Actual', color='blue')
sns.lineplot(data=df_test, x='Day', y='Predicted_PM2.5', label='Predicted', color='red')
plt.title('PM2.5 Prediction (2023-2024) using Decision Tree')
plt.xlabel('Date')
plt.ylabel('PM2.5')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()
