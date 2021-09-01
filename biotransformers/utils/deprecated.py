import functools
import warnings


def deprecated_alias(**aliases):
    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            rename_kwargs(f.__name__, kwargs, aliases)
            return f(*args, **kwargs)

        return wrapper

    return deco


def rename_kwargs(func_name, kwargs, aliases):  # noqa
    for alias, new in aliases.items():
        if alias in kwargs:
            if new in kwargs:
                raise TypeError(
                    "{} received both {} and {}".format(func_name, alias, new)
                )
            warnings.warn(
                "{} is deprecated; use {}".format(alias, new), DeprecationWarning, 3
            )

            if alias == "device":
                if "cuda" in kwargs[alias]:
                    kwargs.pop(alias)
                    kwargs[new] = 1
                elif "cpu" in kwargs[alias]:
                    kwargs.pop(alias)
                    kwargs[new] = 0
                else:
                    kwargs[new] = kwargs.pop(alias)

            elif alias == "multi_gpu":
                kwargs.pop(alias)
            else:
                kwargs[new] = kwargs.pop(alias)
