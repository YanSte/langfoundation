from __future__ import annotations

import logging
from typing import Set

from rich.console import Console

from langfoundation.utils.debug.log.filter import LogFilter
from langfoundation.utils.debug.log.handler import LogHandler
from langfoundation.utils.debug.log.log_level import LogLevel

logger = logging.getLogger(__name__)


def configure_log(
    log_levels: Set[LogLevel] = {
        LogLevel.CRITICAL,
        LogLevel.ERROR,
        LogLevel.ERROR,
        LogLevel.WARNING,
        LogLevel.INFO,
        LogLevel.DEBUG,
    },
) -> None:
    """
    Configure the logging module.

    This function configures the logging module to write logs to the console.
    The log level can be set using the `log_levels` parameter.

    The log levels are defined in the `LogLevel` enum.

    The log format is set to `%(message)s`. This means that only the log message
    is displayed, without any additional information such as the timestamp,
    filename, and line number.

    The log messages are written to the console in red color.

    The log level can be set to one of the following values:

    * `LogLevel.CRITICAL`: Critical errors that make the program unusable.
    * `LogLevel.ERROR`: Errors that make the program behave incorrectly.
    * `LogLevel.WARNING`: Warnings that might indicate a problem.
    * `LogLevel.INFO`: Informational messages that are useful for debugging.
    * `LogLevel.DEBUG`: Debug messages that are useful for developers.

    The default log level is set to `LogLevel.INFO`.

    The log messages are filtered using the `LogFilter` class. The filter
    allows log messages with a level that is in the `allowed_levels` set.

    The log messages are written to the console using the `LogHandler` class.
    The handler is configured to write logs to the console with the
    `console` parameter. The `console` parameter is set to an instance of the
    `Console` class from the `rich` library. The `Console` class is used to
    write text to the console with colors and styles.

    The log messages are written to the console in red color using the
    `error_console` parameter. The `error_console` parameter is set to an
    instance of the `Console` class from the `rich` library. The `Console`
    class is used to write text to the console with colors and styles.

    The `configure_log` function is called automatically when the
    `langfoundation` module is imported. The log level is set to
    `LogLevel.INFO` by default.

    Examples:
        >>> from langfoundation.debug import configure_log
        >>> configure_log({LogLevel.ERROR})
        >>> configure_log({LogLevel.ERROR, LogLevel.WARNING})
        >>> configure_log({LogLevel.ERROR, LogLevel.WARNING, LogLevel.INFO})
    """
    error_console = Console(
        stderr=True,
        width=Console().size.width,
    )

    rh = LogHandler(console=error_console)
    rh.clear_log()
    rh.setFormatter(logging.Formatter("%(message)s"))
    allowed_levels: Set[int] = {level.value for level in log_levels}
    rh.addFilter(LogFilter(allowed_levels))

    logging.basicConfig(
        level=logging.NOTSET,
        handlers=[rh],
    )
