"""Utility functions for NLP research."""
import logging

import coloredlogs

# Set up logging
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('debug.log', mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
file_handler.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(coloredlogs.ColoredFormatter('%(asctime)s %(levelname)s %(message)s'))
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(file_handler)
