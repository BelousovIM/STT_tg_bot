import shutil
from tempfile import NamedTemporaryFile

import uvicorn
from fastapi import FastAPI, UploadFile

from stt import HOST, PORT
from stt.stt_client import stt_client

app = FastAPI()


@app.post("/voices")
async def create_item(files: UploadFile):
    file = files.file
    with NamedTemporaryFile() as tmpfile:
        shutil.copyfileobj(file, tmpfile)
        tmpfile.seek(0)
        response_message = stt_client(
            audio_filename=tmpfile.name, model_name="quartznet15x5_torch",
        )
        return response_message


def main():
    uvicorn.run("stt.main:app", reload=True, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
