from config import Config


def test_config():
    config = Config("config.yaml")
    assert "device" in config
    assert config["device"] == "cuda"
    assert config.device == "cuda"
