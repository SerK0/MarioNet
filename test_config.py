from config import Config


def test_config():
    config = Config(
        {
            "device": "cuda",
            "model": {
                "TargetEncoder": {
                    "param1": "value1",
                    "param2": "value2",
                }
            },
        }
    )
    assert "device" in config
    assert config["device"] == "cuda"
    assert config.device == "cuda"

    assert "model" in config
    assert "TargetEncoder" in config.model
    assert config.model.TargetEncoder.param1 == "value1"
    assert config.model.TargetEncoder.param2 == "value2"


def test_config_from_file():
    config = Config.from_file("config.yaml")
    assert "device" in config
    assert config["device"] == "cuda"
    assert config.device == "cuda"
