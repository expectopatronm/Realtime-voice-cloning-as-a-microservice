from dotenv import load_dotenv

load_dotenv(verbose=True)
from pydantic import BaseSettings


class Settings(BaseSettings):

    BASE_URL = "/service/voice-cloning"


# Instantiate all settings once so it can be imported from other modules
settings = Settings()
