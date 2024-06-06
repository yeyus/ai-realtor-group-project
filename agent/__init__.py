import logging
import sys


logging.basicConfig(
    stream=sys.stderr,
    level=logging.DEBUG,
    format="%(module)s - %(filename)s - %(levelname)s - %(message)s",
)
