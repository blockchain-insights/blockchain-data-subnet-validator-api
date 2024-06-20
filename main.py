from dotenv import load_dotenv
load_dotenv()

from api.insight_api import APIServer

server = APIServer()
server.start()