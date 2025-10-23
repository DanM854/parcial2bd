# test_main.py
import pytest
import pandas as pd
import boto3
from moto import mock_aws
import io
import pyarrow as pa
import pyarrow.parquet as pq
import holidays


@mock_aws
def test_s3_upload_and_schema():
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="sakila-19191919")

    df_fact = pd.DataFrame({
        "rental_date": pd.to_datetime([
            "2005-06-01", "2005-06-02", "2005-06-03",
            "2005-12-25",
        ])
    })

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

    table = pa.Table.from_pandas(df_date, schema=schema)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="snappy")

    s3.put_object(
        Bucket="sakila-19191919",
        Key="dim_date/dim_date.parquet",
        Body=buffer.getvalue()
    )

    obj = s3.get_object(Bucket="sakila-19191919", Key="dim_date/dim_date.parquet")
    data = obj["Body"].read()
    assert len(data) > 0, "El archivo Parquet no se escribió correctamente"

    read_table = pq.read_table(io.BytesIO(data))
    df_result = read_table.to_pandas()

    expected_cols = {
        "date", "date_id", "year", "month", "month_name",
        "day", "day_name", "quarter", "is_weekend", "is_holiday_us"
    }
    assert expected_cols.issubset(df_result.columns), "Faltan columnas esperadas"
    assert df_result["year"].min() == 2005
    assert df_result.loc[df_result["date"].astype(str) == "2005-12-25", "is_holiday_us"].iloc[0] is True


def test_date_id_format():
    sample_date = pd.to_datetime("2005-07-14")
    date_id = int(sample_date.strftime("%Y%m%d"))
    assert date_id == 20050714


def test_weekend_detection():
    dates = pd.to_datetime(["2023-11-18", "2023-11-19", "2023-11-20"])
    df = pd.DataFrame({"date": dates})
    df["is_weekend"] = df["date"].dt.weekday >= 5
    assert df["is_weekend"].tolist() == [True, True, False]
