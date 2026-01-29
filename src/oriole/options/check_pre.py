from ..error import new_error
from .config import Config
from ..util.files import check_parent_dir_exists


def check_prerequisites(config: Config) -> None:
    if config.files.trace:
        check_parent_dir_exists(config.files.trace)
    check_parent_dir_exists(config.files.params)

