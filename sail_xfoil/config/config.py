import configparser
import ast

class Config:
    def __init__(self, config_file):

        print("\n\nenter config.py ...\n\n")

        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        for key, value in self.config['INTEGER'].items():
            print(key,value)
            key = key.upper()
            setattr(self, key, int(value))

        for key, value in self.config['ARRAYS'].items():
            key = key.upper()
            setattr(self, key, ast.literal_eval(value))

        for key, value in self.config['TUPLE_ARRAYS'].items():
            key = key.upper()
            setattr(self, key, ast.literal_eval(value))