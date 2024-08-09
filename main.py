from dotenv import load_dotenv
from api.insight_api import APIServer

load_dotenv()
server = APIServer()
server.start()