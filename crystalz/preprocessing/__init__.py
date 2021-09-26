"""
Modules involved in the preprocessing step
"""

from types import ModuleType

from crystalz.preprocessing import dummy, overlaps


def get_method_name(method_module: ModuleType) -> str:
    """
    FInds the name of a preprocessing method from its implementation module.
    The name is actually the unqualified name of the module.

    Parameters
    ----------
    method_module: ModuleType
        The implementation module

    Returns
    -------
    str
        Name of the method
    """
    return method_module.__name__.split('.')[-1]


_method_modules = [overlaps, dummy]
"""List here all the methods to expert"""


METHODS = {
    get_method_name(module): module
    for module in _method_modules
}
"""Dictionary of exported preprocessing methods"""
