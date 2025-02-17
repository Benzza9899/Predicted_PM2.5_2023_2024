import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

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

# เลือกเฉพาะข้อมูลปี 2020-2024
df = df[(df['Year'] >= 2020) & (df['Year'] <= 2024)]

# ปรับค่า Feature Scaling
scaler = StandardScaler()
df[['Year', 'Month_x', 'TempAVG']] = scaler.fit_transform(df[['Year', 'Month_x', 'TempAVG']])

# แปลงข้อมูลให้เหมาะกับการ Train
X = df[['Year', 'Month_x', 'TempAVG']].values
y = df['PM2.5'].values

# เปรียบเทียบโมเดล
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression (Degree 4)": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Support Vector Regression": SVR(kernel='rbf')
}

results = []

for name, model in models.items():
    if "Polynomial" in name:
        poly = PolynomialFeatures(degree=4)
        X_poly = poly.fit_transform(X)
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
    else:
        model.fit(X, y)
        y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    results.append([name, r2, mse])

# สร้าง DataFrame สำหรับแสดงผลลัพธ์
results_df = pd.DataFrame(results, columns=["Model", "R² Score", "MSE"])
print(results_df)

# ใช้โมเดลที่ดีที่สุด (Random Forest) ทำนายปี 2025
best_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_model.fit(X, y)

# ✅ สร้างข้อมูลปี 2025 ใหม่
date_range_2025 = pd.date_range(start="2025-01-01", periods=365, freq='D')
future_data = pd.DataFrame({
    'Day': date_range_2025,
    'Year': 2025,
    'Month_x': date_range_2025.month
})

# คำนวณค่าเฉลี่ย TempAVG ของแต่ละเดือนจากข้อมูลในอดีต
monthly_avg_temp = df.groupby('Month_x')['TempAVG'].mean()

# เติมค่า TempAVG ให้ข้อมูลปี 2025 ตามค่าเฉลี่ยของเดือนนั้น ๆ
future_data['TempAVG'] = future_data['Month_x'].map(monthly_avg_temp)

# ปรับค่า Feature Scaling
future_data[['Year', 'Month_x', 'TempAVG']] = scaler.transform(future_data[['Year', 'Month_x', 'TempAVG']])

# ทำนายค่า PM2.5
future_data['Predicted_PM2.5'] = best_model.predict(future_data[['Year', 'Month_x', 'TempAVG']])

# รวมข้อมูลเก่ากับข้อมูลทำนาย
df['Data_Type'] = 'Actual'
future_data['Data_Type'] = 'Predicted'
combined_df = pd.concat([
    df[['Day', 'Year', 'Month_x', 'PM2.5', 'Data_Type']],
    future_data[['Day', 'Year', 'Month_x', 'Predicted_PM2.5', 'Data_Type']].rename(columns={'Predicted_PM2.5': 'PM2.5'})
])

# ✅ แสดงข้อมูลท้ายสุดเพื่อตรวจสอบว่าปี 2025 ถูกต้องหรือไม่
print(combined_df[['Day', 'Year', 'PM2.5', 'Data_Type']].tail(10))

# สร้างกราฟ
plt.figure(figsize=(12, 6))
sns.lineplot(data=combined_df, x='Day', y='PM2.5', hue='Data_Type', 
             palette={'Actual': 'blue', 'Predicted': 'red'})

# ปรับแต่งกราฟ
plt.title('PM2.5 Trends: 2020-2025')
plt.xlabel('Date')
plt.ylabel('PM2.5')
plt.legend(title="Data Type", loc="upper right")
plt.xticks(rotation=45)
plt.grid()
plt.xlim(pd.Timestamp("2020-01-01"), pd.Timestamp("2025-12-31"))

# แสดงผลลัพธ์
plt.show()
