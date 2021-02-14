import yaml


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                value = DotDict(value)
            elif isinstance(value, list):
                value = [
                    DotDict(item) if isinstance(item, dict) else item for item in value
                ]

            self[key] = value


class Config(DotDict):
    @classmethod
    def from_file(cls, filepath="config.yaml"):
        with open(filepath, "r") as config_file:
            config_dict = yaml.load(config_file)

        return cls(config_dict)
