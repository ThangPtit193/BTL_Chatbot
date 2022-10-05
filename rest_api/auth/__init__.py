import os
from dotenv import load_dotenv
from rest_api.auth.router import ServicesWrapper

load_dotenv()

AXIOM_EMAIL = os.getenv('AXIOM_EMAIL', '')
AXIOM_PASSWORD = os.getenv('AXIOM_PASSWORD', '')

services_wrapper = ServicesWrapper(
    axiom_email=AXIOM_EMAIL,
    axiom_password=AXIOM_PASSWORD
)

if __name__ == "__main__":
    pass
