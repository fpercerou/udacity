import logging
from datetime import datetime
import os


def get_standard_logger(name, log_dir, log_prefix, level='INFO'):
    """
    Returns a logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # file handler
    time_stamp = datetime.today().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "%s_%s.log" % (log_prefix, time_stamp))
    handler = logging.FileHandler(log_path)
    handler.setLevel(getattr(logging, level))

    # formatting
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s : %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger