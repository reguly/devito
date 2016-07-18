import ctypes
from sympy import symbols
import numpy as np


def convert_dtype_to_ctype(dtype):
    """maps Python types to C types.

    :param dtype: a Python numpy type of int32, float32, int64 or float64.
    :return: corrospondinf C type
    """
    conversion_dict = {np.int32: ctypes.c_int, np.float32: ctypes.c_float,
                       np.int64: ctypes.c_int64, np.float64: ctypes.c_double}
    return conversion_dict[dtype]


def sympy_find(expr, term, repl):
    """chage all terms from function notation to array notation.

    :param expr: the expression to be condsred for changed.
    :param term: the pattern to be replaced.
    :param repl: the pattern replacing.
    :return: the changed expr.

    Finds all terms of the form term(x1, x2, x3)
    and changes them to repl[x1, x2, x3]. i.e. changes from
    function notation to array notation. It also reorders the indices
    x1, x2, x3 so that the time index comes first.
    """

    t = symbols("t")
    if type(expr) == term:
        args_wo_t = [x for x in expr.args if x != t and t not in x.args]
        args_t = [x for x in expr.args if x == t or t in x.args]
        expr = repl[tuple(args_t + args_wo_t)]
    if hasattr(expr, "args"):
        for a in expr.args:
            expr = expr.subs(a, sympy_find(a, term, repl))
    return expr


def aligned(a, alignment=16):
    """Function to align the memmory.

    :param a: the memory given.
    :param alignment: granularity of alignment, 16 bytes by default.
    :return: refeence to the start of the aligned memmory.
    """
    if (a.ctypes.data % alignment) == 0:
        return a

    extra = alignment / a.itemsize
    buf = np.empty(a.size + extra, dtype=a.dtype)
    ofs = (-buf.ctypes.data % alignment) / a.itemsize
    aa = buf[ofs:ofs+a.size].reshape(a.shape)
    np.copyto(aa, a)
    assert (aa.ctypes.data % alignment) == 0
    return aa
