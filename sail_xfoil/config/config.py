import configparser
import ast

class Config:
    def __init__(self, config_file):

        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        for key, value in self.config['INTEGER'].items():
            key = key.upper()
            setattr(self, key, int(value))

        for key, value in self.config['ARRAYS'].items():
            key = key.upper()
            setattr(self, key, ast.literal_eval(value))

        for key, value in self.config['TUPLE_ARRAYS'].items():
            key = key.upper()
            setattr(self, key, ast.literal_eval(value))

        for key, value in self.config['FLOAT'].items():
            key = key.upper()
            setattr(self, key, float(value))