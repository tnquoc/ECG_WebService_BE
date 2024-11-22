from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd


app = FastAPI()  # khởi tạo 1 đối tượng API

data_path = "LargeStudentData.csv"

df = pd.read_csv(data_path)


@app.get("")
def home():
    return {"message": "hello world"}


@app.get("/get_data")
def get_data():

    data = df.to_dict(orient="records")
    return data


@app.get("/get_data_by_student_id")
def get_data(student_id: int = None):

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


@app.get("/get_list_data_by_student_ids/{student_ids}")
def get_list_data(student_ids: str):

    try:
        student_id_list = [int(idx) for idx in student_ids.split(",")]

        filtered_df = df[df["StudentID"].isin(student_id_list)]

        return filtered_df.to_dict(orient="records")  #
    except Exception as e:
        return {"error": f"An error occurred while processing the data: {str(e)}"}


class ListStudentId(BaseModel):
    student_ids: list[int]


@app.post("/post_list_student_id")
def get_list_student_id(liststudentID: ListStudentId):

    try:
        student_ids = liststudentID.student_ids
        df = df[df["StudentID"].isin(student_ids)]

        # Kiểm tra nếu không có dữ liệu phù hợp
        if df.empty:
            return {"message": "No matching student found."}

        data = df.to_dict(orient="records")
        return data
    except Exception as e:
        return {"error": f"An error occurred while processing the data: {str(e)}"}


class StudentInfo(BaseModel):
    student_name: str
    student_class: str


@app.post("/add_student")
def post_student(new_student: StudentInfo):

    try:
        global df
        if df.empty:
            new_id = 1
        else:
            new_id = int(df["StudentID"].max() + 1)

        new_row_student = {
            "StudentID": new_id,
            "StudentName": new_student.student_name,
            "Class": new_student.student_class,
        }

        df = pd.concat([df, pd.DataFrame([new_row_student])], ignore_index=True)

        # Ghi lại vào file CSV
        df.to_csv(data_path, index=False)
        return {"message": "Student added successfully.", "data": new_row_student}

    except Exception as e:
        return {"error": f"An error occurred while processing the data: {str(e)}"}


@app.post("/update_student")
def update_student_info(student_info: StudentInfo):

    try:
        df["StudentID"] = pd.to_numeric(df["StudentID"], errors="coerce")
        filtered_df = df[df["StudentID"] == student_info.student_id]

        if filtered_df.empty:
            return {"error": f"StudentID {student_info.student_id} does not exist."}

        # Lấy chỉ số dòng
        idx = filtered_df.index[0]
        if student_info.student_name:
            df.at[idx, "StudentName"] = student_info.student_name
        if student_info.student_class:
            df.at[idx, "Class"] = student_info.student_class

        # Ghi lại vào file CSV
        df.to_csv(data_path, index=False)
        updated_data = df.loc[idx].to_dict()
        return {"message": "Student added successfully.", "data": updated_data}

    except Exception as e:
        return {"error": f"An error occurred while processing the data: {str(e)}"}
