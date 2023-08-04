import configparser

class Config:
    def __init__(self, config_file):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        # Bind all config values as attributes of the class
        for key, value in self.config['DEFAULT'].items():
            setattr(self, key, value)