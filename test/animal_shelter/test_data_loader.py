from animal_shelter import data_loader
import logging

LOG = logging.getLogger(__name__)

def test_convert_camel_case():
    LOG.info("asd as asd")
    assert data_loader.convert_camel_case("CamelCase") == "camel_case"
    assert data_loader.convert_camel_case("CamelCASE") == "camel_case"
    assert data_loader.convert_camel_case("camel-case") == "camel-case"
