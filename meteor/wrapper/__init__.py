# import os
# from dotenv import load_dotenv
# from logging import Logger
#
# from meteor.wrapper.axiom_client import AxiomModelWrapper
#
# load_dotenv()
#
# _logger = Logger(__name__)
#
# AXIOM_EMAIL = os.getenv('AXIOM_EMAIL', '')
# AXIOM_PASSWORD = os.getenv('AXIOM_PASSWORD', '')
# HUB_ID = os.getenv('HUB_ID', 347)
# HUB_VERSION = os.getenv('HUB_VERSION', 'v0.1.0')
# MODELS_ID = os.getenv('MODELS_ID', 187)
# MODELS_VERSION = os.getenv('MODELS_VERSION', 'v0.1')
#
# axiom_wrapper = AxiomModelWrapper(
#     axiom_email=AXIOM_EMAIL,
#     axiom_password=AXIOM_PASSWORD,
#     model_ids=MODELS_ID,
#     model_version=MODELS_VERSION,
#     hub_id=HUB_ID,
#     hub_version=HUB_VERSION
# )
#
