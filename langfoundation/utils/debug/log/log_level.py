from enum import Enum


class LogLevel(Enum):
    """
    Enum for log levels.

    The log levels are defined as follows:

    * CRITICAL: Critical errors that make the program unusable.
    * ERROR: Errors that make the program behave incorrectly.
    * WARNING: Warnings that might indicate a problem.
    * INFO: Informational messages that are useful for debugging.
    * DEBUG: Debug messages that are useful for developers.
    * NOTSET: The default log level, which is the lowest level.
    """

    CRITICAL = 50
    FATAL = 50
    ERROR = 40
    WARNING = 30
    WARN = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0
