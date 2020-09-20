import configparser as cp

class SimpleParserMain:
    def __main__():
        parser = cp.RawConfigParser()
        parser.read_file('settings.config')
        print(parser.get('simple_parser', 'learning_grade')
