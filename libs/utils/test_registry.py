test_func_registry = {}

def register_test_func(*names):
    def decorator(cls):
        for name in names:
            if isinstance(name, (list, tuple)):
                for n in name:
                    test_func_registry[n] = cls
            else:
                test_func_registry[name] = cls
        return cls

    return decorator

def get_test_func(name):
    """
         A simple test function getter
    """
    test_func = test_func_registry[name]
    return test_func