import sys
import argparse

import torch

class LogEntry(object):

    def __init__(self, init_losses=[], init_metrics=[]):
        assert isinstance(init_losses, list)
        self._losses = {}
        for key in init_losses:
            self._losses[key] = 0

        assert isinstance(init_metrics, list)
        self._metrics = {}
        for key in init_metrics:
            self._metrics[key] = 0

    def __repr__(self):
        return str(self)

    def __str__(self):
        loss_str = ''
        for key, value in self._losses.items():
            if value != 0:
                loss_str += ' | {}: {:.4f}'.format(key, value)
        loss_str += ' | {}: {:.4f}'.format('Total', sum(self._losses.values()))

        metric_str = ''
        for key, value in self._metrics.items():
            if value != 0:
                metric_str += ' | {}: {:.4f}'.format(key, value)

        out = ''
        if len(loss_str) > 0:
            out += 'Losses\t{}\n'.format(loss_str)
        if len(metric_str) > 0:
            out += 'Metrics\t{}\n'.format(metric_str)

        if len(out) > 0:
            out = out[:-1]

        return out

    @property
    def losses(self):
        return self._losses

    @property
    def metrics(self):
        return self._metrics

    def clear(self):
        self._losses = {}
        self._metrics = {}

    def reset(self):
        for key in self._losses:
            self._losses[key] = 0
        for key in self._metrics:
            self._metrics[key] = 0

    def add_loss(self, key):
        if key not in self._losses:
            self._losses[key] = 0

    def add_metric(self, key):
        if key not in self._metrics:
            self._metrics[key] = 0

    def itemize(self):
        for key, value in self._losses.items():
            if isinstance(value, torch.Tensor):
                self._losses[key] = value.item()

        for key, value in self._metrics.items():
            if isinstance(value, torch.Tensor):
                self._metrics[key] = value.item()

    def absorb(self, other_log):
        assert isinstance(other_log, LogEntry)

        for key, value in other_log.losses.items():
            if key in self._losses:
                self._losses[key] += value
            else:
                self._losses[key] = value

        for key, value in other_log.metrics.items():
            if key in self._metrics:
                self._metrics[key] += value
            else:
                self._metrics[key] = value

    def average(self, N):
        for key in self._losses:
            self._losses[key] /= N

        for key in self._metrics:
            self._metrics[key] /= N

    def to_dict(self):
        return {'losses': self._losses, 'metrics': self.metrics}


class Logger(object):
    """
    Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file.
    """

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace