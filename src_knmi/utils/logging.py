import logging

def setup_logging(level=logging.DEBUG):
    logger = logging.getLogger('meteosaver')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    logging.basicConfig(level=level, filename='debug.log',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        force=True, filemode='w')

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(f'logs/{level}.log')
    file_handler.setLevel(level)

    # Create a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger