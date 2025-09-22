import os
from enum import Enum
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()  # take environment variables


class NautobotEnv(str, Enum):
    LOCAL = "local"
    NONPROD = "nonprod"
    PROD = "prod"


class Credential(TypedDict):
    NAUTOBOT_URL: str
    NAUTOBOT_TOKEN: str


class NautobotCredentialMapping:
    @staticmethod
    def get_credentials(env: NautobotEnv) -> Credential:
        # Here you would pull the correct credentials based on env
        if env == NautobotEnv.LOCAL:
            return {
                "NAUTOBOT_URL": "http://localhost:8080/",
                "NAUTOBOT_TOKEN": os.environ.get(
                    "NAUTOBOT_LOCAL_TOKEN", "0123456789abcdef0123456789abcdef01234567"
                ),
            }
        elif env == NautobotEnv.NONPROD:
            return {
                "NAUTOBOT_URL": os.environ.get("NAUTOBOT_NONPROD_BASE_URL", ""),
                "NAUTOBOT_TOKEN": os.environ.get("NAUTOBOT_NONPROD_TOKEN", ""),
            }
        elif env == NautobotEnv.PROD:
            return {
                "NAUTOBOT_URL": os.environ.get("NAUTOBOT_PROD_BASE_URL", ""),
                "NAUTOBOT_TOKEN": os.environ.get("NAUTOBOT_PROD_TOKEN", ""),
            }
        else:
            raise ValueError(f"Unknown environment: {env}")
