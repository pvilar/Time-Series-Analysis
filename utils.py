""" Helper functions """

def set_logger(name: str = "timeseries") -> logging.Logger:
    """
    Returns a formatted logger to use in scripts.

    Args
      name: The name of the logger

    Returns
      logger: A logging.Logger object

    """
    logger = logging.getLogger(name)
    formatter = logging.Formatter(name + " - %(asctime)s - %(message)s", "%H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger