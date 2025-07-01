import csv
from pathlib import Path
from models import FinancialScore
import pandas as pd
from dotenv import load_dotenv
import os
import io
import boto3

load_dotenv()
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
KEY = os.getenv("aws_access_key_id")
SECRET = os.getenv("aws_secret_access_key")

def save_score_to_csv(score_data: FinancialScore, output_path="scores.csv"):
    # Convertimos el dict a DataFrame transpuesto
    df = pd.DataFrame.from_dict(score_data.dict(), orient="index", columns=["valor"])
    df.index.name = "indicador"

    df.to_csv(output_path)

    s3 = boto3.client('s3',aws_access_key_id=KEY, aws_secret_access_key=SECRET)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer)
    s3.put_object(Bucket=BUCKET_NAME, Key=output_path, Body=csv_buffer.getvalue())

def read_csv_from_s3(filename: str) -> pd.DataFrame:
    try:
        s3 = boto3.client('s3',aws_access_key_id=KEY, aws_secret_access_key=SECRET)
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=filename)
        return pd.read_csv(obj["Body"])
    except Exception as e:
        print(f"error en descargar files: ")
        raise Exception()

def obtener_margen(score):
    if score > 80:
        return 0.20
    elif score > 60:
        return 0.35
    else:
        return 0.50

def ajuste_por_volumen(vol):
    if vol > 1000:
        return -0.05
    elif vol > 500:
        return -0.02
    else:
        return 0.0