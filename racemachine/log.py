import logging
import racemachine.config as config

levels = {
        'DEBUG': logging.DEBUG,
        'INFO':  logging.INFO,
        'WARN':  logging.WARN,
        'ERROR': logging.ERROR
        }

LEVEL = config.get('log.level')
LEVEL = LEVEL if LEVEL else 'INFO'
LEVEL = levels[LEVEL]

FORMAT = '%(asctime)-15s %(message)s'

logging.basicConfig(level=LEVEL, format=FORMAT)

def get_logger(name):
    return logging.getLogger(name)
