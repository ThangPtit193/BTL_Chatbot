import inspect
import logging
import os
import sys
from datetime import datetime as dt
from typing import List, Text, Any

import coloredlogs


def get_bool_env(var):
    var_value = os.getenv(var, None)
    if var_value is None or var_value in ['0', 'False', 'FALSE']:
        return False
    else:
        return True


def get_name_ony_fr_path(file, _all_exts=None):
    """
    Get name onlu from path.
    Ex: /root/name/name.txt --> name
        /root/name/name/    --> name
        name.txt            --> name
        name                --> name

    :param file: file path
    :param _all_exts: extenstion that u want to remove
    :return:
    """
    all_exts = ['.jpg', '.json', '.png', '.txt', '.ini', '.bmp']
    if _all_exts is not None:
        all_exts += _all_exts

    file = file.strip('/')
    texts = file.split('/')
    name_file = texts[-1]
    for ext in all_exts:
        name_file = name_file.replace(ext, '')
    return name_file


class EndpointFilter(logging.Filter):
    def __init__(
            self,
            path: str,
            *args: Any,
            **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._path = path

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find(self._path) == -1


class LogTracker:
    def __init__(self, file_name=None, reset=False, isPrintLog=True, showOneLevel=False):
        """

        Args:
            file_name:
            reset:
            isPrintLog:
            showOneLevel:
        """
        if file_name is not None:
            self.__logPath = file_name
            self.__isWritetLog = True
            if not os.path.isfile(file_name):
                os.system('touch ' + file_name)
        else:
            self.__logPath = None
            self.__isWritetLog = False

        if self.__logPath and reset and os.path.isfile(self.__logPath):
            os.system('rm -rf ' + self.__logPath)
        self.__show_one_level = showOneLevel
        self.__isPrintLog = isPrintLog
        self.__prePrintLog = []
        self.__preWriteLog = []

    def set_show_one_level(self, value):
        self.__show_one_level = value

    def enable_print_log(self):
        self.__isPrintLog = True

    def disable_print_log(self):
        self.__isPrintLog = False

    def enable_write_log(self):
        self.__isWritetLog = True

    def disable_write_log(self):
        self.__isWritetLog = False

    def reset(self):
        os.system('rm -rf ' + self.__logPath)

    def log(self, level, msg, show=True, has_prefix=True):
        """
        trace message
        Args:
            level:
            msg:
            show:
            has_prefix:

        Returns:

        """
        return self._log(level, msg, show=show, has_prefix=has_prefix)

    def _log(self, level, _msg, has_prefix=True, show=True):
        msg = _msg
        target_text = ''
        level_name = logging.getLevelName(level)
        if not has_prefix:
            prefix = ''
        else:
            prefix = f"{self._get_asctime()}  {level_name}  " \
                     f"{get_name_ony_fr_path(self._get_filename(), ['.py'])}:" \
                     f"{self._get_lineno()}: "
        if self.__isPrintLog:
            target_text = "{}{}".format(prefix, msg)
            if show:
                print("{}{}".format(prefix, msg))
        if self.__isWritetLog:
            with open(self.__logPath, 'a') as f:
                f.write("{}{}\n".format(prefix, _msg))
        return target_text

    def _get_asctime(self):
        return dt.now().strftime('[%Y%m%d-%H:%M:%S:%f]')

    def _get_filename(self):
        stack = inspect.stack()
        return stack[4][1]

    def _get_funcName(self):
        stack = inspect.stack()
        return stack[4][3]

    def _get_lineno(self):
        stack = inspect.stack()
        return stack[4][2]

    def holdPrintLog(self, isPrintLog):
        self.__prePrintLog.append(self.__isPrintLog)
        self.__isPrintLog = isPrintLog

    def releasePrintLog(self):
        if len(self.__prePrintLog) == 0:
            raise Exception('Nothing to release')
        self.__isPrintLog = self.__prePrintLog.pop()

    def holdWriteLog(self, isWriteLog):
        self.__preWriteLog.append(self.__isWritetLog)
        self.__isWritetLog = isWriteLog

    def releaseWriteLog(self):
        if len(self.__preWriteLog) == 0:
            raise Exception('Nothing to release')
        self.__isWritetLog = self.__preWriteLog.pop()


log_tracker = LogTracker(file_name=None, reset=True, isPrintLog=False)


class LEVEL(object):
    CRITICAL = logging.CRITICAL
    FATAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET


class COLOR(object):
    RED = "red"
    BLACK = "black"
    GREEN = "green"
    ORANGE = "orange"
    BLUE = "blue"
    PURPLE = "purple"
    CYAN = "cyan"
    LIGHT_GRAY = "lightgrey"
    DARK_GRAY = "darkgrey"
    LIGHT_GREEN = "lightgreen"
    YELLOW = "yellow"
    PINK = "pink"
    FAIL = "fail"


# STYLE
class STYLE(object):
    BOLD = "bold"
    DISABLE = "disable"
    UNDERLINE = "underline"
    STRIKE_THROUGH = "strikethrough"
    INVISIBLE = "invisible"
    REVERSE = "reverse"
    RESET = "reset"


class PrintColor(object):
    def __init__(self):
        self.attrs = {
            'bold': '\033[01m',
            'disable': '\033[02m',
            'underline': '\033[04m',
            'reverse': '\033[07m',
            'strikethrough': '\033[09m',
            'invisible': '\033[08m',
            "reset": '\033[0m'

        }
        self.colors = {
            "fail": '\033[91m',
            "black": '\033[30m',
            "red": '\033[31m',
            "green": '\033[32m',
            "orange": '\033[33m',
            "blue": '\033[34m',
            "purple": '\033[35m',
            "cyan": '\033[36m',
            "lightgrey": '\033[37m',
            "darkgrey": '\033[90m',
            "lightred": '\033[91m',
            "lightgreen": '\033[92m',
            "yellow": '\033[93m',
            "lightblue": '\033[94m',
            "pink": '\033[95m',
            "lightcyan": '\033[96m',
        }
        self.on_colors = {
            "black": '\033[40m',
            "red": '\033[41m',
            "green": '\033[42m',
            "orange": '\033[43m',
            "blue": '\033[44m',
            "purple": '\033[45m',
            "cyan": '\033[46m',
            "lightgrey": '\033[47m'
        }

    def print(self, msg, color: Text = None, on_color: Text = None, attrs: List = None,
              show: bool = True):
        """
        Print color

        Args:
            msg: The message will be print out
            color: color accepted:
            on_color: background color
            attrs: a dictionary containing:
                bold:
                disable:
                underline:
                reverse:
                strikethrough:
                invisible
                reset:
            show: show this message into terminal: default True

        Returns:

        """
        if color is not None:
            if color not in self.colors.keys():
                print(f"{self.__class__.__name__} do not support color {color}")
            else:
                msg = "{}{}".format(self.colors[color], msg)
        if on_color is not None:
            if color not in self.colors.keys():
                print(f"{self.__class__.__name__} do not support onclor: {on_color}")
            else:
                msg = "{}{}".format(self.on_colors[on_color], msg)
        if attrs is not None:
            for style in attrs:
                if style not in self.attrs.keys():
                    print(f"{self.__class__.__name__} do not support {style} style")
                    continue
                msg = "{}{}".format(self.attrs[style], msg)
        if color is not None or on_color is not None or \
                attrs is not None:
            msg = "{}{}".format(msg, self.attrs['reset'])
        if show: print(msg)
        return msg


_print_color = PrintColor()


def _set_colorlogs_env():
    os.environ[
        # "COLOREDLOGS_LOG_FORMAT"] = "%(asctime)s %(name)s %(levelname)5.5s   %(module)12.12s:%(lineno)3.3s - %(message)s"
        "COLOREDLOGS_LOG_FORMAT"] = "%(asctime)s %(levelname)5.5s %(module)16.16s:%(lineno)3.3s - %(message)s"
    os.environ['COLOREDLOGS_FIELD_STYLES'] = ';'.join([
        'asctime=green',
        'levelname=black,bold',
        'funcName=blue',
        'module=blue'
    ])
    os.environ['COLOREDLOGS_LEVEL_STYLES'] = ';'.join([
        'info=white'
        'debug=white',
        'warning=yellow',
        'success=118,bold',
        'error=red',
        'critical=background=red'])
    os.environ['COLOREDLOGS_DATE_FORMAT'] = '%Y-%m-%d %H:%M:%S'


def configure_logger(level: Text = None):
    """
    Configure  log

    Returns:

    """
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    # logging.getLogger("transitions.core").setLevel(logging.WARNING)
    logging.getLogger("denver").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    logging.getLogger("allennlp").setLevel(logging.WARNING)
    logging.getLogger("sentry_sdk.errors").setLevel(logging.WARNING)

    # Disable logger fastAPI
    uvicorn_logger = get_logger("uvicorn.access")
    uvicorn_logger.addFilter(EndpointFilter(path="/api/version"))

    # DIsable logger for sentry
    sentry_logger = get_logger("sentry_sdk.internal")
    sentry_logger.addFilter(EndpointFilter(path="/api/74"))

    _set_colorlogs_env()
    level = os.environ.get("LOG_LEVEL", "INFO") if not level else level
    level = level.upper()
    coloredlogs.install(level=level)
    # logging.basicConfig(filename="log.log", level=logging.DEBUG)


if hasattr(sys, '_getframe'):
    currentframe = lambda: sys._getframe(3)
    _srcfile = os.path.normcase(currentframe.__code__.co_filename)
else:
    _srcfile = ""


def findCallerPatch(*args, **kwargs):
    """
    Find the stack frame of the caller so that we can note the source
    file name, line number and function name.

    Args:
        stack_info:

    Returns:

    """
    f = currentframe()
    if f is not None:
        f = f.f_back
    rv = "(unknown file)", 0, "(unknown function)"
    while hasattr(f, "f_code"):
        co = f.f_code
        filename = os.path.normcase(co.co_filename)
        if filename == _srcfile:
            f = f.f_back
            continue
        rv = (co.co_filename, f.f_lineno, co.co_name, None)
        break
    return rv


def isEnabledFor(level: int):
    """
    Is this logger enabled for level 'level'?
    """
    global_level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO"))
    return (level == global_level) or (level >= logging.INFO)


def get_logger(name_logger):
    """
    Create a child logger specific to a module'

    Args:
        name_logger:

    Returns:

    """

    module_logger = logging.getLogger(name_logger)
    # module_logger.findCaller = findCallerPatch

    if os.environ.get("LOG_MONO", False):
        module_logger.isEnabledFor = isEnabledFor

    def pr(msg: Text, color: Text = None, on_color: Text = None):
        _print_color.print(msg, color, on_color)

    def nl(msg, color=COLOR.PURPLE):
        if logging.root.level < LEVEL.NL:
            return _print_color.print(msg, color=color)

    def trace(msg, show=True, has_prefix=True, level=logging.INFO, *args, **kwargs):
        log_tracker.log(level, msg, has_prefix=has_prefix, show=False)
        if show:
            return module_logger.log(level, msg, *args, **kwargs)

    setattr(module_logger, 'pr', pr)
    setattr(module_logger, 'nl', nl)
    setattr(module_logger, 'trace', trace)
    return module_logger


if __name__ == "__main__":
    configure_logger()
    _logger = get_logger(__name__)
    _logger.info("this a message")
    _logger.debug("this a message")
    _logger.warning("this a message")
    _logger.error("this a message")
