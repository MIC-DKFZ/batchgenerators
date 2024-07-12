import logging

# Create a logger to use instead of root logger
logger = logging.getLogger('batchgen')
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
log = logger.log # Useful as it can be imported by other files