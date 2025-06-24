# ECG_WebService_BE

## Run Locally

```
pip install -r requirements.txt
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Run with pre-built Docker Image

```
docker pull quoctn/ecg_be:latest
docker run -d -p 8008:8000 quoctn/ecg_be:latest
```

## Integrate with Frontend

Link to frontend repo: https://github.com/tnquoc/ECG_WebService_FE

## Contact

Reach address tnquoc1998@gmail.com for other supports.
