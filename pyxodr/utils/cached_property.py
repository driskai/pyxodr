try:
    from functools import cached_property
except ImportError:
    from functools import lru_cache

    def cached_property(func):
        """Redefine functools' cached_property decorator for Python <=3.7."""

        @property
        @lru_cache()
        def cached_func(*args, **kwargs):
            return func(*args, **kwargs)

        return cached_func
