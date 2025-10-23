import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import io
import holidays


def create_dim_date(df_fact):
    """
    Crea un DataFrame con la dimensión de fechas basado en 'rental_date'.
    """
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
    return df_date


def test_dim_date_structure():
    df_fact = pd.DataFrame({
        "rental_date": pd.to_datetime([
            "2005-06-01", "2005-06-02", "2005-06-03",
            "2005-12-25"  # Navidad
        ])
    })

    df_date = create_dim_date(df_fact)

    expected_cols = {
        "date", "date_id", "year", "month", "month_name",
        "day", "day_name", "quarter", "is_weekend", "is_holiday_us"
    }

    assert expected_cols.issubset(df_date.columns), "Faltan columnas esperadas"
    assert df_date["year"].nunique() == 1 and df_date["year"].iloc[0] == 2005
    assert df_date.loc[df_date["date"].astype(str) == "2005-12-25", "is_holiday_us"].iloc[0] is True


def test_date_id_format():
    sample_date = pd.to_datetime("2005-07-14")
    date_id = int(sample_date.strftime("%Y%m%d"))
    assert date_id == 20050714, "El formato de date_id es incorrecto"


def test_weekend_detection():
    dates = pd.to_datetime(["2023-11-18", "2023-11-19", "2023-11-20"])
    df = pd.DataFrame({"date": dates})
    df["is_weekend"] = df["date"].dt.weekday >= 5
    assert df["is_weekend"].tolist() == [True, True, False], "Error en detección de fin de semana"


def test_parquet_roundtrip():
    """
    Verifica que los datos puedan guardarse y leerse como Parquet correctamente.
    """
    df_fact = pd.DataFrame({"rental_date": pd.to_datetime(["2005-06-01", "2005-06-02"])})
    df_date = create_dim_date(df_fact)

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

    buffer = io.BytesIO()
    table = pa.Table.from_pandas(df_date, schema=schema)
    pq.write_table(table, buffer, compression="snappy")

    buffer.seek(0)
    df_loaded = pq.read_table(buffer).to_pandas()

    assert df_loaded.equals(df_date), "Los datos leídos no coinciden con los escritos"
