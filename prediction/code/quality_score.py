import os
import sqlite3
import pandas as pd
from pathlib import Path

def process_fdb_file(fdb_file):
    try:
        conn = sqlite3.connect(f"file:{fdb_file}?mode=ro", uri=True)
        cur = conn.cursor()

        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cur.fetchall()

        table_name = 'WELL'
        cur.execute(f"SELECT * FROM {table_name}")
        column_names = [desc[0] for desc in cur.description]
        records = cur.fetchall()
        df = pd.DataFrame(records, columns=column_names)

        required_cols = ['ID', 'Type', 'Par', 'Well', 'Time', 'Val']
        if all(col in df.columns for col in required_cols):
            df = df[df['Par'].isin(['TE', 'ICM'])].copy()
            df['Event'] = df['Par']
            df['RowName'] = df['ID'].astype(str) + '_' + df['Well'].astype(str)
            matrix = df.pivot_table(index='RowName', columns='Event', values='Val', aggfunc='first')

            def classify_quality(row):
                icm = str(row.get('ICM', '')).strip()
                te = str(row.get('TE', '')).strip()
                score = icm + te
                return "good" if score in {"AA", "AB", "BA"} else "fair"

            matrix['Quality'] = matrix.apply(classify_quality, axis=1)
            matrix['SourceFile'] = Path(fdb_file).name
            return matrix
        else:
            return None

    except Exception as e:
        return None
    finally:
        cur.close()
        conn.close()

def extract_patient_birthdates(patient_db_files):
    birthdate_records = []
    for fdb_file in patient_db_files:
        try:
            conn = sqlite3.connect(f"file:{fdb_file}?mode=ro", uri=True)
            cur = conn.cursor()

            cur.execute("SELECT * FROM IDTable")
            id_table = cur.fetchall()
            id_df = pd.DataFrame(id_table, columns=["PatientIDX", "PatientID", "Name", "Time", "Unknown"])

            cur.execute("SELECT * FROM Patients")
            patients_table = cur.fetchall()
            patients_df = pd.DataFrame(patients_table, columns=["PatientIDX", "Par", "Val", "Time", "Unknown"])

            dob_df = patients_df[patients_df["Par"] == "DateOfBirth"].copy()
            dob_df = dob_df[["PatientIDX", "Val"]].rename(columns={"Val": "DateOfBirth"})
            dob_df["DateOfBirth"] = pd.to_datetime(pd.to_numeric(dob_df["DateOfBirth"], errors="coerce"), origin="1899-12-30", unit="D")

            merged = pd.merge(id_df[["PatientIDX", "PatientID"]], dob_df, on="PatientIDX", how="inner")
            merged["SourceFile"] = Path(fdb_file).name
            birthdate_records.append(merged)

        except Exception as e:
            pass
        finally:
            cur.close()
            conn.close()

    if birthdate_records:
        return pd.concat(birthdate_records)[["PatientIDX", "PatientID", "DateOfBirth", "SourceFile"]]
    else:
        return pd.DataFrame(columns=["PatientIDX", "PatientID", "DateOfBirth", "SourceFile"])

if __name__ == '__main__':
    fdb_files = [
        Path("/Users/lzz/Downloads/MainDB_5068.fdb"),
        Path("/Users/lzz/Downloads/MainDB_5069.fdb"),
        Path("/Users/lzz/Downloads/MainDB_4424.fdb"),
        Path("/Users/lzz/Downloads/MainDB_4425.fdb"),
    ]

    patients_db_files = [
        Path("/Users/lzz/Downloads/PatientsDB_5068.fdb"),
        Path("/Users/lzz/Downloads/PatientsDB_5069.fdb"),
        Path("/Users/lzz/Downloads/PatientsDB_4424.fdb"),
        Path("/Users/lzz/Downloads/PatientsDB_4425.fdb"),
    ]

    all_matrices = []
    for fdb_file in fdb_files:
        matrix = process_fdb_file(fdb_file)
        if matrix is not None:
            all_matrices.append(matrix)

    if all_matrices:
        merged = pd.concat(all_matrices)
        merged.to_csv("./MainDB.csv")
    else:
        pass

    birth_df = extract_patient_birthdates(patients_db_files)
    birth_df.to_csv("./PatientBirthDates.csv", index=False)

    # Extract SlideID-PatientIDX mapping from PatientsDB files
    def extract_slide_patientidx_from_patients_db(patient_db_files):
        slide_records = []
        for fdb_file in patient_db_files:
            try:
                conn = sqlite3.connect(f"file:{fdb_file}?mode=ro", uri=True)
                df = pd.read_sql_query("SELECT * FROM SlideID", conn)
                df.columns = ["SlideID", "PatientIDX"]
                df["SourceFile"] = Path(fdb_file).name
                slide_records.append(df)
            except Exception as e:
                pass
            finally:
                conn.close()
        if slide_records:
            return pd.concat(slide_records, ignore_index=True)
        else:
            return pd.DataFrame(columns=["SlideID", "PatientIDX", "SourceFile"])

    slide_df = extract_slide_patientidx_from_patients_db(patients_db_files)
    merged_df = pd.merge(slide_df, birth_df.rename(columns={"SourceFile": "PatientDB_File"}), on="PatientIDX", how="left")
    # Defer column filtering until after fertilization data merge

    def extract_fertilization_times_from_main_fdbs(fdb_files):
        fert_records = []
        for fdb_file in fdb_files:
            try:
                conn = sqlite3.connect(f"file:{fdb_file}?mode=ro", uri=True)
                df = pd.read_sql_query("SELECT * FROM GENERAL", conn)
                df.columns = ["ID", "Type", "Par", "Time1", "Val", "Time2", "Unknown"]  # match actual table structure
                df = df[df["Par"] == "Fertilization"].copy()
                df["FertilizationTime"] = pd.to_datetime(pd.to_numeric(df["Val"], errors="coerce"), origin="1899-12-30", unit="D").dt.date
                df["PatientID"] = df["ID"].astype(str)
                df["SourceFile"] = Path(fdb_file).name
                fert_records.append(df[["PatientID", "FertilizationTime", "SourceFile"]])
            except Exception as e:
                pass
            finally:
                conn.close()
        if fert_records:
            return pd.concat(fert_records, ignore_index=True)
        else:
            return pd.DataFrame(columns=["PatientID", "FertilizationTime", "SourceFile"])

    fert_df = extract_fertilization_times_from_main_fdbs(fdb_files)

    merged_df["PatientID"] = merged_df["PatientID"].astype(str).str.strip()
    fert_df["PatientID"] = fert_df["PatientID"].astype(str).str.strip()

    # merged_df 中的 SlideID 实际对应的是 fert_df 中的 PatientID
    merged_df = pd.merge(merged_df, fert_df, left_on="SlideID", right_on="PatientID", how="left")

    merged_test = pd.merge(
        merged_df[["SlideID"]].rename(columns={"SlideID": "PatientID"}).drop_duplicates(),
        fert_df,
        on="PatientID",
        how="left"
    )

    source_col = "SourceFile_x" if "SourceFile_x" in merged_df.columns else ("SourceFile" if "SourceFile" in merged_df.columns else "PatientDB_File")
    merged_df = merged_df.rename(columns={source_col: "SourceFile"})
    merged_df = merged_df[["SlideID", "PatientIDX", "PatientID_y", "DateOfBirth", "FertilizationTime", "SourceFile"]]
    merged_df = merged_df.rename(columns={"PatientID_y": "PatientID"})

    merged_df["Age"] = (pd.to_datetime(merged_df["FertilizationTime"]) - pd.to_datetime(merged_df["DateOfBirth"])).dt.days / 365.25
    merged_df["Age"] = merged_df["Age"].round().astype("Int64")

    merged_df = merged_df[["SlideID", "PatientIDX", "PatientID", "DateOfBirth", "FertilizationTime", "Age", "SourceFile"]]
    merged_df.to_csv("./SlidePatientBirthDates.csv", index=False)