import inspect, logging
import os.path
from six.moves import cPickle
from blocks.extensions import SimpleExtension
from blocks.serialization import secure_dump

logger = logging.getLogger(__name__)

SAVED_TO = "saved_to"


class SaveTheBest(SimpleExtension):
    """Check if a log quantity has the minimum/maximum value so far.
    If so, save the model.

    Parameters
    ----------
    record_name : str
        The name of the record to track.
    notification_name : str, optional
        The name for the record to be made in the log when the current
        value of the tracked quantity is the best so far. It not given,
        'record_name' plus "best_so_far" suffix is used.
    choose_best : callable, optional
        A function that takes the current value and the best so far
        and return the best of two. By default :func:`min`, which
        corresponds to tracking the minimum value.
    path : str
        File to save the model as.

    Attributes
    ----------
    best_name : str
        The name of the status record to keep the best value so far.
    notification_name : str
        The name of the record written to the log when the current
        value of the tracked quantity is the best so far.

    Notes
    -----
    In the likely case that you are relying on another extension to
    add the tracked quantity to the log, make sure to place this
    extension *after* the extension that writes the quantity to the log
    in the `extensions` argument to :class:`blocks.main_loop.MainLoop`.

    """
    def __init__(self, record_name, path, notification_name=None,
                 choose_best=min,
                 save_separately=None,
                 use_cpickle=False, **kwargs):
        self.record_name = record_name
        if not notification_name:
            notification_name = record_name + "_best_so_far"
        self.notification_name = notification_name
        self.best_name = "best_" + record_name
        self.choose_best = choose_best
        kwargs.setdefault("after_epoch", True)
        super(SaveTheBest, self).__init__(**kwargs)
        if not save_separately:
            save_separately = []
        self.path = path
        self.save_separately = save_separately
        self.use_cpickle = use_cpickle
    
    def save_separately_filenames(self, path):
        root, ext = os.path.splitext(path)
        return {attribute: root + "_" + attribute + ext
                for attribute in self.save_separately}

    def do(self, callback_name, *args):
        current_value = self.main_loop.log.current_row.get(self.record_name)
        if current_value is None:
            return
        best_value = self.main_loop.status.get(self.best_name, None)
        if (best_value is None or
                (current_value != best_value and
                 self.choose_best(current_value, best_value) ==
                 current_value)):
            self.main_loop.status[self.best_name] = current_value
            self.main_loop.log.current_row[self.notification_name] = True
            _, from_user = self.parse_args(callback_name, args)
            try:
                path = self.path
                if from_user:
                    path, = from_user
                    secure_dump(self.main_loop, path, use_cpickle=self.use_cpickle)
                    filenames = self.save_separately_filenames(path)
                    for attribute in self.save_separately:
                        secure_dump(getattr(self.main_loop, attribute),
                                filenames[attribute], cPickle.dump)
            except Exception:
                path = None
                raise
            finally:
                already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
                self.main_loop.log.current_row[SAVED_TO] = (already_saved_to +
                                                            (path,))
