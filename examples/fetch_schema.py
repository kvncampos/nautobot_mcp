import json

import requests

from utils.config import config


class SchemaFetcher:
    """This class fetches and saves the OpenAPI schema from a specified URL."""

    def __init__(self):
        self.token = "0123456789abcdef0123456789abcdef01234567"

    def fetch_and_save(self, url, output_file):
        headers = {"Authorization": f"Token {self.token}"}
        response = requests.get(url, headers=headers, verify=config.SSL_VERIFY)
        response.raise_for_status()
        with open(output_file, "w") as f:
            json.dump(response.json(), f, indent=2)
            print(f"Schema saved to {output_file}")


if __name__ == "__main__":
    # Replace 'your_token_here' with your actual token
    url = "http://localhost:8080/api/swagger.json"
    output_file = "examples/openapi_schema.json"
    fetcher = SchemaFetcher()
    fetcher.fetch_and_save(url, output_file)
