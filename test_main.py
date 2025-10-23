import pytest
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO
import holidays
from datetime import date
import boto3
from moto import mock_s3  # Para simular AWS S3

# Importar funciones si tu script las separas en un módulo
# from dim_date_etl import generar_dim_date

# ------------------------------
# 1. Simular datos de entrada
# ------------------------------

@pytest.fixture
def sample_fact_rental_df():
    data = {
        "rental_date": [
            "2005-05-24",
            "2005-05-25",
            "2005-05-28",  # Sábado
            "2005-07-04"   # Festivo US
        ]
    }
    return pd.DataFrame(data)

# ------------------------------
# 2. Prueba: generación de dim_date
# ------------------------------

def test_dim_date_generation(sample_fact_rental_df):
    # Simular el proceso principal sin escribir en S3
    df_fact = sample_fact_rental_df

    # Transformación igual a la del script
    df_date = pd.DataFrame(df_fact["rental_date"].dropna().unique(), columns=["date"])
    df_date["date"] = pd.to_datetime(df_date["date"]).dt.date
    df_date = df_date.sort_values("date").reset_index(drop=True)

    us_holidays = holidays.US()
    df_date["date_id"] = pd.to_datetime(df_date["date"]).dt.strftime("%Y%m%d").astype(int)
    df_date["year"] = pd.to_datetime(df_date["date"]).dt.year
    df_date["month"] = pd.to_datetime(df_date["date"]).dt.month
    df_date["month_name"] = pd.to_datetime(df_date["date"]).dt.strftime("%B")
    df_date["day"] = pd.to_datetime(df_date["date"]).dt.day
    df_date["day_name"] = pd.to_datetime(df_date["date"]).dt.strftime("%A")
    df_date["quarter"] = "Q" + pd.to_datetime(df_date["date"]).dt.quarter.astype(str)
    df_date["is_weekend"] = pd.to_datetime(df_date["date"]).dt.weekday >= 5
    df_date["is_holiday_us"] = df_date["date"].isin(us_holidays)

    # Validar columnas
    expected_columns = [
        "date", "date_id", "year", "month", "month_name",
        "day", "day_name", "quarter", "is_weekend", "is_holiday_us"
    ]
    assert list(df_date.columns) == expected_columns

    # Validar que las fechas son correctas
    assert df_date.iloc[0]["date"] == date(2005, 5, 24)
    assert df_date.iloc[1]["day_name"] == "Wednesday"

    # Validar date_id
    assert df_date[df_date["date"] == date(2005, 5, 24)]["date_id"].iloc[0] == 20050524

    # Validar fin de semana
    assert df_date[df_date["date"] == date(2005, 5, 28)]["is_weekend"].iloc[0] is True

    # Validar festivo US (4 de julio)
    assert df_date[df_date["date"] == date(2005, 7, 4)]["is_holiday_us"].iloc[0] is True


# ------------------------------
# 3. Prueba: error si falta rental_date
# ------------------------------

def test_error_no_rental_date():
    df_fact = pd.DataFrame({"otra_columna": [1, 2, 3]})
    with pytest.raises(KeyError):
        _ = df_fact["rental_date"]


# ------------------------------
# 4. Prueba: escritura simulada en S3
# ------------------------------

@mock_s3
def test_write_to_s3(sample_fact_rental_df):
    import boto3
    s3 = boto3.client("s3", region_name="us-east-1")
    bucket = "sakila-test-bucket"
    s3.create_bucket(Bucket=bucket)

    # Crear parquet simulado
    buffer = BytesIO()
    table = pa.Table.from_pandas(sample_fact_rental_df)
    pq.write_table(table, buffer)
    s3.put_object(Bucket=bucket, Key="fact_rental/fact_rental.parquet", Body=buffer.getvalue())

    # Leer de nuevo para verificar
    response = s3.get_object(Bucket=bucket, Key="fact_rental/fact_rental.parquet")
    body = BytesIO(response["Body"].read())
    df_loaded = pq.read_table(body).to_pandas()

    assert not df_loaded.empty
    assert "rental_date" in df_loaded.columns
