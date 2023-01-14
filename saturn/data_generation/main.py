from saturn.data_generation.tripple_generator import TripleGenerator
from comet.lib import logger

logger.configure_logger("DEBUG")
_logger = logger.get_logger(__name__)
if __name__ == '__main__':
    triple_generator = TripleGenerator("config/config_timi.yaml")
    triple_generator.load()
    _logger.info("Load data")

    triple_generator.generate_triples()
    # triple_generator.generate_quadruplet()
