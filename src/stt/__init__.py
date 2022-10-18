import pathlib

project_root = pathlib.Path(__file__).resolve().parent.parent.parent
data_path = project_root / "data"

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"
BOT_TOKEN = "5546752267:AAHg2pzmwdnZ2PsTkFPFPvpgpriQ3wh1yY8"
