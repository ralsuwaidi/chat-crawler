import logging
from datetime import datetime  # Corrected import statement
import os

# Get the current datetime
now = datetime.now()

# Format the datetime string
logger_name = now.strftime("%Y%m%d_%H%M%S")

# Create logger object
logger = logging.getLogger(logger_name)  # Use the logger_name instead of 'my_logger'
logger.setLevel(logging.DEBUG)

# Create file handler
if not os.path.exists("logs"):
    os.mkdir("logs")
fh = logging.FileHandler("logs/" + logger_name + ".log")
fh.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(fh)
