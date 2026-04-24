import asyncio
import os
import time
import threading
from datetime import datetime
from typing import List, Optional, Tuple, Union, Dict

import pandas as pd


async def load_data(path: str) -> pd.DataFrame:
    """Асинхронно загружает данные из Excel файла."""

    df = await asyncio.to_thread(pd.read_excel, path, engine='openpyxl')
    df.columns = df.columns.str.strip().str.lower()

    return df


async def load_multiple_files(files: List[str]) -> pd.DataFrame:
    """Параллельно загружает список файлов и объединяет их."""

    tasks = [load_data(file) for file in files]
    dfs = await asyncio.gather(*tasks)

    return pd.concat(dfs, ignore_index=True)


def parse_date(value: Union[str, datetime, float]) -> Optional[datetime]:
    """Преобразует строку с датой в объект datetime."""

    if pd.isna(value):
        return None

    value = str(value).strip()
    formats = ["%Y-%m-%d", "%d.%m.%Y", "%b %d, %Y", "%B %d, %Y"]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            pass

    return None


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Преобразует колонки с датами."""

    date_columns = [
        "install_date",
        "warranty_until",
        "last_calibration_date",
        "last_service_date"
    ]

    for col in date_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_date)

    return df


def normalize_status_column(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит статусы оборудования к стандартным категориям."""

    mapping = {
        "ok": "operational", "working": "operational", "op": "operational",
        "maintenance": "maintenance_scheduled", "broken": "faulty", "error": "faulty"
    }

    df["status"] = df["status"].apply(
        lambda x: mapping.get(str(x).strip().lower(), str(x).lower()) if pd.notna(x) else "unknown"
    )

    return df


def clean_uptime(df: pd.DataFrame) -> pd.DataFrame:
    """Преобразует проценты аптайма в числовой формат."""

    df["uptime_pct"] = pd.to_numeric(
        df["uptime_pct"].astype(str).str.replace(",", "."),
        errors="coerce"
    )

    return df


def check_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Проверяет логическую корректность дат калибровки."""

    mask = (df["install_date"].notna()) & (df["last_calibration_date"].notna())
    invalid = mask & (df["last_calibration_date"] < df["install_date"])
    df.loc[invalid, "last_calibration_date"] = None

    return df


def filter_warranty_task(df: pd.DataFrame, results: Dict):
    """Фильтрует устройства по статусу гарантии."""

    today = datetime.today()
    results['in_w'] = df[df["warranty_until"] >= today]
    results['out_w'] = df[df["warranty_until"] < today]


def clinics_problems_task(df: pd.DataFrame, results: Dict):
    """Агрегирует количество проблем по клиникам."""

    results['clinics'] = (
        df.groupby(["clinic_id", "clinic_name"])
        .agg({"issues_reported_12mo": "sum"})
        .sort_values("issues_reported_12mo", ascending=False)
    )


def calibration_report_task(df: pd.DataFrame, results: Dict):
    """Готовит данные для отчета по калибровке."""

    results['calib'] = df[[
        "device_id", "clinic_name", "model", "last_calibration_date"
    ]]


def summary_table_task(df: pd.DataFrame, results: Dict):
    """Создает сводную таблицу по клиникам и моделям."""

    results['summary'] = pd.pivot_table(
        df, index=["clinic_name", "model"],
        values=["issues_reported_12mo", "uptime_pct"],
        aggfunc={"issues_reported_12mo": "sum", "uptime_pct": "mean"}
    )


async def save_excel_async(df: pd.DataFrame, folder: str, name: str) -> None:
    """Асинхронно сохраняет DataFrame в файл."""

    path = os.path.join(folder, name)
    await asyncio.to_thread(df.to_excel, path)


async def async_main(files: List[str], output_folder: str) -> None:
    """Асинхронная реализация выполнения задания."""

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    start_time = time.time()

    df = await load_multiple_files(files)

    df = convert_dates(df)
    df = normalize_status_column(df)
    df = clean_uptime(df)
    df = check_dates(df)

    res = {}

    await asyncio.gather(
        asyncio.to_thread(filter_warranty_task, df, res),
        asyncio.to_thread(clinics_problems_task, df, res),
        asyncio.to_thread(calibration_report_task, df, res),
        asyncio.to_thread(summary_table_task, df, res)
    )

    await asyncio.gather(
        save_excel_async(res['in_w'], output_folder, "in_warranty.xlsx"),
        save_excel_async(res['out_w'], output_folder, "out_warranty.xlsx"),
        save_excel_async(res['clinics'], output_folder, "clinics.xlsx"),
        save_excel_async(res['calib'], output_folder, "calibration.xlsx"),
        save_excel_async(res['summary'], output_folder, "summary.xlsx")
    )

    print(f"АСИНХРОННО: {round(time.time() - start_time, 2)} сек")


def threading_main(files: List[str], output_folder: str) -> None:
    """Многопоточная реализация выполнения задания."""

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    start_time = time.time()

    all_dfs = []

    def read_worker(f):
        data = pd.read_excel(f, engine='openpyxl')
        data.columns = data.columns.str.strip().str.lower()
        all_dfs.append(data)

    read_threads = [threading.Thread(target=read_worker, args=(f,)) for f in files]

    for t in read_threads:
        t.start()

    for t in read_threads:
        t.join()

    df = pd.concat(all_dfs, ignore_index=True)
    df = convert_dates(df)
    df = normalize_status_column(df)
    df = clean_uptime(df)
    df = check_dates(df)

    res = {}

    proc_threads = [
        threading.Thread(target=filter_warranty_task, args=(df, res)),
        threading.Thread(target=clinics_problems_task, args=(df, res)),
        threading.Thread(target=calibration_report_task, args=(df, res)),
        threading.Thread(target=summary_table_task, args=(df, res))
    ]

    for t in proc_threads:
        t.start()

    for t in proc_threads:
        t.join()

    def save_worker(data, name):
        data.to_excel(os.path.join(output_folder, name))

    save_map = {
        "in_warranty.xlsx": res['in_w'],
        "out_warranty.xlsx": res['out_w'],
        "clinics.xlsx": res['clinics'],
        "calibration.xlsx": res['calib'],
        "summary.xlsx": res['summary']
    }

    save_threads = [
        threading.Thread(target=save_worker, args=(d, n))
        for n, d in save_map.items()
    ]

    for t in save_threads:
        t.start()

    for t in save_threads:
        t.join()

    print(f"В ПОТОКАХ: {round(time.time() - start_time, 2)} сек")


if __name__ == "__main__":
    file_list = [f"medical_diagnostic_devices_{i}.xlsx" for i in range(1, 11)]
    existing_files = [
        f for f in file_list
        if os.path.exists(f) and os.path.getsize(f) > 0
    ]

    if existing_files:
        asyncio.run(async_main(existing_files, "results_async"))
        threading_main(existing_files, "results_threads")
    else:
        print("Ошибка: Файлы для обработки не найдены.")