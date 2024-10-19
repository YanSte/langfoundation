from __future__ import annotations

import logging
import os
import traceback

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.style import Style
from rich.text import Text


class LogHandler(RichHandler):
    def get_level_style(self, level: int) -> str:
        match level:
            case logging.DEBUG:
                return "yellow3"
            case logging.INFO:
                return "blue"
            case logging.WARNING:
                return "bold orange"
            case logging.ERROR:
                return "bold red"
            case logging.CRITICAL:
                return "bold magenta"
            case _:
                return "white"

    def clear_log(self) -> None:
        if os.path.exists("logs.log"):
            os.remove("logs.log")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            has_title = hasattr(record, "title")
            title = "[ " + getattr(record, "title", f"{record.filename} : {record.funcName}") + " ]"

            match record.levelno:
                case logging.ERROR:
                    console_capture = Console(width=Console().size.width)
                    with console_capture.capture() as capture:
                        message = self.format(record)
                        msg = record.msg if has_title else message
                        console_capture.print(msg)
                    str_output = capture.get()
                    str_output = f"In: {record.pathname}:{record.lineno}\n\n{str_output}"
                    # Add stack trace to the output if exc_info is present
                    if record.exc_info:
                        stack_trace = "".join(traceback.format_exception(None, record.exc_info[1], record.exc_info[2]))
                        str_output += f"\n\nStack Trace:\n{stack_trace}"

                    text = Text.from_ansi(str_output)

                    box_style = Style(
                        color="white",
                        blink=True,
                        bgcolor="red",
                    )
                    panel = Panel(
                        text,
                        title=f"[bold]{title}[/]",
                        padding=(1, 2),
                        style=box_style,
                    )
                    self.console.print(panel)

                case logging.INFO:
                    console_capture = Console()
                    with console_capture.capture() as capture:
                        message = self.format(record)
                        msg = record.msg if has_title else message
                        console_capture.print(msg)
                    str_output = capture.get()
                    text = Text.from_ansi(str_output)

                    panel = Panel(
                        text,
                        title=title,
                        padding=(1, 2),
                        style="none",
                    )
                    self.console.print(panel)

                case _:
                    message = self.format(record)
                    msg = record.msg if has_title else message
                    self.console.print(msg)

        except Exception:
            message = self.format(record)
            msg = record.msg if has_title else message
            text = Text(msg)
            box_style = Style(
                color="white",
                blink=True,
                bgcolor="red",
            )
            panel = Panel(
                text,
                title=f"LOG EXCEPTION ERROR in {record.filename}: {record.funcName}",
                padding=(1, 2),
                style=box_style,
            )
            self.console.print(panel)
