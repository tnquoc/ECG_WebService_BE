from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd


app = FastAPI()  # khởi tạo 1 đối tượng API


@app.get("/")
def home():
    return {"message": "hello world"}


@app.get("/get_data")
def get_data():
    data_path = "LargeStudentData.csv"

    df = pd.read_csv(data_path)
    data = df.to_dict(orient="records")
    return data


@app.get("/get_data_by_student_id")
def get_data(student_id: int = None):
    data_path = "LargeStudentData.csv"

    df = pd.read_csv(data_path)

    try:
        # kiểm tra url có studentID hay không, nếu có thì chỉ chọn những cột dữ liệu  mà cột StudentID khớp với student_id có trong url
        if student_id is not None:
            df = df[df["StudentID"] == student_id]

        # Kiểm tra nếu không có dữ liệu phù hợp
        if df.empty:
            return {"message": "No matching student found."}

        data = df.to_dict(orient="records")
        return data
    except Exception as e:
        return {"error": f"An error occurred while processing the data: {str(e)}"}


@app.get("/get_list_data_by_student_id/")
def get_list_data(student_id: list[int] = Query(default=[])):
    data_path = "LargeStudentData.csv"

    df = pd.read_csv(data_path)

    try:
        # kiểm tra url có studentID hay không, nếu có thì chỉ chọn những cột dữ liệu  mà cột StudentID khớp với student_id có trong url

        df = df[df["StudentID"].isin(student_id)]

        # Kiểm tra nếu không có dữ liệu phù hợp
        if df.empty:
            return {"message": "No matching student found."}

        data = df.to_dict(orient="records")
        return data
    except Exception as e:
        return {"error": f"An error occurred while processing the data: {str(e)}"}


@app.get("/get_data_by_student_ids/")
def get_list_data(student_ids: str = Query(default="")):
    data_path = "LargeStudentData.csv"

    df = pd.read_csv(data_path)

    try:
        # kiểm tra url có studentID hay không, nếu có thì chỉ chọn những cột dữ liệu  mà cột StudentID khớp với student_id có trong url
        if not student_ids:
            return {"message": "No student IDs provided."}

        student_id_list = [int(sid) for sid in student_ids.split(",")]

        df = df[df["StudentID"].isin(student_id_list)]

        # Kiểm tra nếu không có dữ liệu phù hợp
        if df.empty:
            return {"message": "No matching student found."}

        data = df.to_dict(orient="records")
        return data
    except Exception as e:
        return {"error": f"An error occurred while processing the data: {str(e)}"}


class StudentInfo1(BaseModel):
    student_id: list[int]


@app.post("/post_list_student_id/")
def post_list_studentid(studentID: StudentInfo1):
    data_path = "LargeStudentData.csv"

    df = pd.read_csv(data_path)

    try:
        student_ids = studentID.student_id
        df = df[df["StudentID"].isin(student_ids)]

        # Kiểm tra nếu không có dữ liệu phù hợp
        if df.empty:
            return {"message": "No matching student found."}

        data = df.to_dict(orient="records")
        return data
    except Exception as e:
        return {"error": f"An error occurred while processing the data: {str(e)}"}


class StudentInfo2(BaseModel):
    student_id: int
    student_name: str
    student_class: str


@app.post("/add_student")
def post_student(new_student: StudentInfo2):
    data_path = "LargeStudentData.csv"

    df = pd.read_csv(data_path)

    try:
        new_row_student = {
            "StudentID": new_student.student_id,
            "StudentName": new_student.student_name,
            "Class": new_student.student_class,
        }
        df = pd.concat([df, pd.DataFrame([new_row_student])], ignore_index=True)

        # Ghi lại vào file CSV
        df.to_csv(data_path, index=False)
        return {"message": "Student added successfully.", "data": new_row_student}

    except Exception as e:
        return {"error": f"An error occurred while processing the data: {str(e)}"}


class updaterStudentInfo(BaseModel):
    student_id: int
    student_name: str
    student_class: str


@app.post("/update_student")
def post_student(updated_student: updaterStudentInfo):
    data_path = "LargeStudentData.csv"

    df = pd.read_csv(data_path)

    try:
        df["StudentID"] = pd.to_numeric(df["StudentID"], errors="coerce")
        filtered_df = df[df["StudentID"] == updated_student.student_id]

        if filtered_df.empty:
            return {"error": f"StudentID {updated_student.student_id} does not exist."}

        # Lấy chỉ số dòng
        idx = filtered_df.index[0]
        if updated_student.student_name:
            df.at[idx, "StudentName"] = updated_student.student_name
        if updated_student.student_class:
            df.at[idx, "Class"] = updated_student.student_class

        # Ghi lại vào file CSV
        df.to_csv(data_path, index=False)
        updated_data = df.loc[idx].to_dict()
        return {"message": "Student added successfully.", "data": updated_data}

    except Exception as e:
        return {"error": f"An error occurred while processing the data: {str(e)}"}
