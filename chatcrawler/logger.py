import logging

# Create logger object
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# Create file handler
fh = logging.FileHandler('my_log.log')
fh.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(fh)
