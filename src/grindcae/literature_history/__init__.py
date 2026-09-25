"""Literature force coupled to the retained fixed-mesh history engine."""
from .models import LiteratureHistoryCase, MODE, RESULT_FORMAT


def run_literature_history(*args, **kwargs):
    from .workflow import run_literature_history as run
    return run(*args, **kwargs)


def read_result(*args, **kwargs):
    from .workflow import read_result as read
    return read(*args, **kwargs)
