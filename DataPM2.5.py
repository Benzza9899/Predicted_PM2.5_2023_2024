import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# โหลดข้อมูลจากไฟล์ Excel
file_path = "F:\PM2.5 DATASET\คำนวณเปรียบเทียบปริมาณฝุ่น PM 2.5 กับปริมาณการสูบบุหรี่ .xlsx"
xls = pd.ExcelFile(file_path)

# รวมข้อมูลจากทุกปีเป็น df_years
df_list = []
for sheet in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet)
    df_list.append(df)

df_years = pd.concat(df_list, ignore_index=True)

# ตรวจสอบว่าคอลัมน์ที่ต้องใช้มีอยู่จริง
required_columns = {'Year', 'Day', 'PM2.5'}
if not required_columns.issubset(df_years.columns):
    raise ValueError(f"Missing required columns: {required_columns - set(df_years.columns)}")

# สร้างตัวแปรอิสระ (X) และตัวแปรตาม (y)
X = df_years[['Year', 'Day']]  
y = df_years['PM2.5']

# สร้างโมเดล Linear Regression
model = LinearRegression()
model.fit(X, y)

# สร้างข้อมูลของปี 2025
days = np.arange(1, 366)  # วันที่ 1-365
year_2025 = np.full_like(days, 2025)

# รวมข้อมูลเป็น DataFrame
df_2025 = pd.DataFrame({'Year': year_2025, 'Day': days})

# ทำการพยากรณ์ค่า PM2.5
df_2025['Predicted_PM2.5'] = model.predict(df_2025)

# แสดงผลลัพธ์
print(df_2025.head())
