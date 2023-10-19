from datetime import datetime
import logging
import logging.config
from pathlib import Path
from config import settings

# Configure external library logging
external_logs = ["requests", "urllib3"]
for log in external_logs:
    logging.getLogger(log).setLevel(logging.CRITICAL)

log_levels = {
    "DEBUG": logging.DEBUG,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class ColorFormatter(logging.Formatter):
    color_map = {
        logging.DEBUG: "\x1b[1;30m",
        logging.INFO: "\x1b[0;37m",
        logging.WARNING: "\x1b[1;33m",
        logging.ERROR: "\x1b[1;31m",
        logging.CRITICAL: "\x1b[1;35m",
    }
    reset = "\x1b[0m"
    _format = "%(asctime)s - [%(threadName)-12.12s] [%(levelname)s-5.5s] -  %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"

    def format(self, record):
        formatter = logging.Formatter(f"{self.color_map.get(record.levelno)}{self._format}{self.reset}")
        return formatter.format(record)


def create_file_logger(filename: str, log_level: str = settings.LOG_LEVEL, log_dir: str = settings.LOG_DIR):
    log_filename = f"{filename}_{datetime.utcnow().astimezone().strftime('%Y-%m-%dT%H-%M-%S')}"
    Path(log_dir).mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger(log_filename)
    logger.setLevel(log_levels.get(log_level, logging.DEBUG))

    file_handler = logging.FileHandler(Path(log_dir) / f"{log_filename}.log")
    file_handler.setFormatter(ColorFormatter())
    logger.addHandler(file_handler)
    logger.debug(f"{log_filename} logger initialized!!!")

    return logger


def init_loggers(log_dir: str = settings.LOG_DIR):
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    logging.config.fileConfig(
        Path(__file__).resolve().parents[1] / "config" / "logging.ini",
        disable_existing_loggers=False,
        defaults={"logdir": log_dir},
    )


def get_logger(log_name: str, log_level: str = settings.LOG_LEVEL):
    logger = logging.getLogger(log_name)
    logger.setLevel(log_levels.get(log_level, logging.DEBUG))
    return logger
