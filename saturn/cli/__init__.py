from comet.constants import env
from comet.lib import logger
import os

_logger = logger.get_logger(__name__)
AXIOM_EMAIL = env.str('AXIOM_EMAIL', '')
AXIOM_PASSWORD = env.str('AXIOM_PASSWORD', '')


def login_axiom():
    from axiom_client.client import Axiom
    client = Axiom()
    if AXIOM_EMAIL and AXIOM_PASSWORD:
        try:
            os.system(f"axiom login --email {AXIOM_EMAIL} --password {AXIOM_PASSWORD}")
            _logger.info(f"Login axiom successfully")
        except Exception as e:
            _logger.warning(f"cannot login axiom: {e}")
