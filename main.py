import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import holidays
import boto3
import s3fs
from io import BytesIO

# === Parámetros configurables ===
input_path = "s3://sakila-19191919/fact_rental/"    
output_path = "s3://sakila-19191919/dim_date/"
bucket_name = "sakila-19191919"                      
output_key = "dim_date/dim_date.parquet"

# === Leer los datos Parquet existentes desde fact_rental ===
print("Leyendo datos desde fact_rental...")
df_fact = pd.read_parquet(input_path)

# Asegurarnos de que rental_date exista
if "rental_date" not in df_fact.columns:
    raise ValueError("No se encontró la columna 'rental_date' en fact_rental")

# === Extraer fechas únicas ===
df_date = pd.DataFrame(df_fact["rental_date"].dropna().unique(), columns=["date"])
df_date["date"] = pd.to_datetime(df_date["date"]).dt.date
df_date = df_date.sort_values("date").reset_index(drop=True)

# === Crear campos derivados ===
us_holidays = holidays.US()
df_date["date_id"] = pd.to_datetime(df_date["date"]).dt.strftime("%Y%m%d").astype(int)
df_date["year"] = pd.to_datetime(df_date["date"]).dt.year
df_date["month"] = pd.to_datetime(df_date["date"]).dt.month
df_date["month_name"] = pd.to_datetime(df_date["date"]).dt.strftime("%B")
df_date["day"] = pd.to_datetime(df_date["date"]).dt.day
df_date["day_name"] = pd.to_datetime(df_date["date"]).dt.strftime("%A")
df_date["quarter"] = "Q" + pd.to_datetime(df_date["date"]).dt.quarter.astype(str)
df_date["is_weekend"] = pd.to_datetime(df_date["date"]).dt.weekday < 5
df_date["is_holiday_us"] = df_date["date"].isin(us_holidays)

# === Definir esquema Parquet con tipo DATE correcto ===
schema = pa.schema([
    pa.field("date", pa.date32()),
    pa.field("date_id", pa.int32()),
    pa.field("year", pa.int16()),
    pa.field("month", pa.int8()),
    pa.field("month_name", pa.string()),
    pa.field("day", pa.int8()),
    pa.field("day_name", pa.string()),
    pa.field("quarter", pa.string()),
    pa.field("is_weekend", pa.bool_()),
    pa.field("is_holiday_us", pa.bool_())
])

# === Guardar en Parquet ===
table = pa.Table.from_pandas(df_date, schema=schema)
buffer = BytesIO()
pq.write_table(table, buffer, compression="snappy")

s3 = boto3.client("s3")
s3.put_object(Bucket=bucket_name, Key=output_key, Body=buffer.getvalue())

