import json
import sys
from pathlib import Path

import requests

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from helpers.creds_helper import NautobotCredentialMapping, NautobotEnv
from utils.config import config


class SchemaFetcher:
    """This class fetches and saves the OpenAPI schema from a specified URL."""

    def __init__(self, token: str = "0123456789abcdef0123456789abcdef01234567"):
        self.token = token

    def fetch_and_save(self, url, output_file):
        headers = {"Authorization": f"Token {self.token}"}
        response = requests.get(url, headers=headers, verify=config.SSL_VERIFY)
        response.raise_for_status()
        with open(output_file, "w") as f:
            json.dump(response.json(), f, indent=2)
            print(f"Schema saved to {output_file}")


if __name__ == "__main__":
    # Change this to whatever environment you want to use
    environment = NautobotEnv.LOCAL

    # Get credentials for the selected environment
    credentials = NautobotCredentialMapping.get_credentials(environment)

    # Build the URL - ensure it has proper protocol and format
    base_url = credentials["NAUTOBOT_URL"].rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = f"https://{base_url}"
    url = f"{base_url}/api/swagger.json"

    output_file = "examples/openapi_schema.json"

    # Create fetcher with the environment's token
    fetcher = SchemaFetcher(token=credentials["NAUTOBOT_TOKEN"])
    fetcher.fetch_and_save(url, output_file)
