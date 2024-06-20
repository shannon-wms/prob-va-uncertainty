"""
Modified functions from scipy.integrate._quad_vec for faster integration.

Author: Shannon Williams

Date: 25/10/2023
"""
import sys
import copy
import heapq
import collections
import functools

import numpy as np

from scipy._lib._util import MapWrapper, _FunctionWrapper
from scipy.integrate._quad_vec import (_max_norm, _get_sizeof, LRUDict, _Bunch)

def _subdivide_interval(args):
    interval, f, norm_func, _quadrature = args
    old_err, a, b, old_int = interval

    c = 0.5 * (a + b)

    # Left-hand side
    if getattr(_quadrature, 'cache_size', 0) > 0:
        f = functools.lru_cache(_quadrature.cache_size)(f)

    _, _, s1, err1, round1 = _quadrature((a, c, f, norm_func))
    dneval = _quadrature.num_eval
    _, _, s2, err2, round2 = _quadrature((c, b, f, norm_func))
    dneval += _quadrature.num_eval
    if old_int is None:
        _, _, old_int, _, _ = _quadrature((a, b, f, norm_func))
        dneval += _quadrature.num_eval

    if getattr(_quadrature, 'cache_size', 0) > 0:
        dneval = f.cache_info().misses

    dint = s1 + s2 - old_int
    derr = err1 + err2 - old_err
    dround_err = round1 + round2

    subintervals = ((a, c, s1, err1), (c, b, s2, err2))
    return dint, derr, dround_err, subintervals, dneval


def _quadrature_trapezoid(args):
    a, b, f, norm_func = args
    """
    Composite trapezoid quadrature
    """
    c = 0.5*(a + b)
    f1 = f(a)
    f2 = f(b)
    f3 = f(c)

    s2 = 0.25 * (b - a) * (f1 + 2*f3 + f2)

    round_err = 0.25 * abs(b - a) * (float(norm_func(f1))
                                       + 2*float(norm_func(f3))
                                       + float(norm_func(f2))) * 2e-16

    s1 = 0.5 * (b - a) * (f1 + f2)
    err = 1/3 * float(norm_func(s1 - s2))
    return a, b, s2, err, round_err


_quadrature_trapezoid.cache_size = 3 * 3
_quadrature_trapezoid.num_eval = 3


def _quadrature_gk(a, b, f, norm_func, x, w, v):
    """
    Generic Gauss-Kronrod quadrature
    """

    fv = [0.0]*len(x)

    c = 0.5 * (a + b)
    h = 0.5 * (b - a)

    # Gauss-Kronrod
    s_k = 0.0
    s_k_abs = 0.0
    for i in range(len(x)):
        ff = f(c + h*x[i])
        fv[i] = ff

        vv = v[i]

        # \int f(x)
        s_k += vv * ff
        # \int |f(x)|
        s_k_abs += vv * abs(ff)

    # Gauss
    s_g = 0.0
    for i in range(len(w)):
        s_g += w[i] * fv[2*i + 1]

    # Quadrature of abs-deviation from average
    s_k_dabs = 0.0
    y0 = s_k / 2.0
    for i in range(len(x)):
        # \int |f(x) - y0|
        s_k_dabs += v[i] * abs(fv[i] - y0)

    # Use similar error estimation as quadpack
    err = float(norm_func((s_k - s_g) * h))
    dabs = float(norm_func(s_k_dabs * h))
    if dabs != 0 and err != 0:
        err = dabs * min(1.0, (200 * err / dabs)**1.5)

    eps = sys.float_info.epsilon
    round_err = float(norm_func(50 * eps * h * s_k_abs))

    if round_err > sys.float_info.min:
        err = max(err, round_err)

    return a, b, h * s_k, err, round_err


def _quadrature_gk21(args):
    a, b, f, norm_func = args
    """
    Gauss-Kronrod 21 quadrature with error estimate
    """
    # Gauss-Kronrod points
    x = (0.995657163025808080735527280689003,
         0.973906528517171720077964012084452,
         0.930157491355708226001207180059508,
         0.865063366688984510732096688423493,
         0.780817726586416897063717578345042,
         0.679409568299024406234327365114874,
         0.562757134668604683339000099272694,
         0.433395394129247190799265943165784,
         0.294392862701460198131126603103866,
         0.148874338981631210884826001129720,
         0,
         -0.148874338981631210884826001129720,
         -0.294392862701460198131126603103866,
         -0.433395394129247190799265943165784,
         -0.562757134668604683339000099272694,
         -0.679409568299024406234327365114874,
         -0.780817726586416897063717578345042,
         -0.865063366688984510732096688423493,
         -0.930157491355708226001207180059508,
         -0.973906528517171720077964012084452,
         -0.995657163025808080735527280689003)

    # 10-point weights
    w = (0.066671344308688137593568809893332,
         0.149451349150580593145776339657697,
         0.219086362515982043995534934228163,
         0.269266719309996355091226921569469,
         0.295524224714752870173892994651338,
         0.295524224714752870173892994651338,
         0.269266719309996355091226921569469,
         0.219086362515982043995534934228163,
         0.149451349150580593145776339657697,
         0.066671344308688137593568809893332)

    # 21-point weights
    v = (0.011694638867371874278064396062192,
         0.032558162307964727478818972459390,
         0.054755896574351996031381300244580,
         0.075039674810919952767043140916190,
         0.093125454583697605535065465083366,
         0.109387158802297641899210590325805,
         0.123491976262065851077958109831074,
         0.134709217311473325928054001771707,
         0.142775938577060080797094273138717,
         0.147739104901338491374841515972068,
         0.149445554002916905664936468389821,
         0.147739104901338491374841515972068,
         0.142775938577060080797094273138717,
         0.134709217311473325928054001771707,
         0.123491976262065851077958109831074,
         0.109387158802297641899210590325805,
         0.093125454583697605535065465083366,
         0.075039674810919952767043140916190,
         0.054755896574351996031381300244580,
         0.032558162307964727478818972459390,
         0.011694638867371874278064396062192)

    return _quadrature_gk(a, b, f, norm_func, x, w, v)


_quadrature_gk21.num_eval = 21


def _quadrature_gk15(args):
    a, b, f, norm_func = args
    """
    Gauss-Kronrod 15 quadrature with error estimate
    """
    # Gauss-Kronrod points
    x = (0.991455371120812639206854697526329,
         0.949107912342758524526189684047851,
         0.864864423359769072789712788640926,
         0.741531185599394439863864773280788,
         0.586087235467691130294144838258730,
         0.405845151377397166906606412076961,
         0.207784955007898467600689403773245,
         0.000000000000000000000000000000000,
         -0.207784955007898467600689403773245,
         -0.405845151377397166906606412076961,
         -0.586087235467691130294144838258730,
         -0.741531185599394439863864773280788,
         -0.864864423359769072789712788640926,
         -0.949107912342758524526189684047851,
         -0.991455371120812639206854697526329)

    # 7-point weights
    w = (0.129484966168869693270611432679082,
         0.279705391489276667901467771423780,
         0.381830050505118944950369775488975,
         0.417959183673469387755102040816327,
         0.381830050505118944950369775488975,
         0.279705391489276667901467771423780,
         0.129484966168869693270611432679082)

    # 15-point weights
    v = (0.022935322010529224963732008058970,
         0.063092092629978553290700663189204,
         0.104790010322250183839876322541518,
         0.140653259715525918745189590510238,
         0.169004726639267902826583426598550,
         0.190350578064785409913256402421014,
         0.204432940075298892414161999234649,
         0.209482141084727828012999174891714,
         0.204432940075298892414161999234649,
         0.190350578064785409913256402421014,
         0.169004726639267902826583426598550,
         0.140653259715525918745189590510238,
         0.104790010322250183839876322541518,
         0.063092092629978553290700663189204,
         0.022935322010529224963732008058970)

    return _quadrature_gk(a, b, f, norm_func, x, w, v)


_quadrature_gk15.num_eval = 15


def quad_vec(f, a, b, epsabs = 1e-200, epsrel = 1e-8, norm = '2', 
             cache_size = 100e6, limit = 10000, workers = 1, points = None, 
             quadrature = None, full_output = False, min_intervals = 2, *, args=()):
    a = float(a)
    b = float(b)
    
    if args:
        if not isinstance(args, tuple):
            args = (args,)
        # create a wrapped function to allow the use of map and Pool.map
        f = _FunctionWrapper(f, args)

    norm_funcs = {
        None: _max_norm,
        'max': _max_norm,
        '2': np.linalg.norm
    }
    if callable(norm):
        norm_func = norm
    else:
        norm_func = norm_funcs[norm]

    parallel_count = 128

    try:
        _quadrature = {None: _quadrature_gk21,
                       'gk21': _quadrature_gk21,
                       'gk15': _quadrature_gk15,
                       'trapz': _quadrature_trapezoid,  # alias for backcompat
                       'trapezoid': _quadrature_trapezoid}[quadrature]
    except KeyError as e:
        raise ValueError(f"unknown quadrature {quadrature!r}") from e

    # Initial interval set
    if points is None:
        initial_intervals = [(a, b)]
    else:
        prev = a
        initial_intervals = []
        for p in sorted(points):
            p = float(p)
            if not (a < p < b) or p == prev:
                continue
            initial_intervals.append((prev, p))
            prev = p
        initial_intervals.append((prev, b))

    global_integral = None
    global_error = None
    rounding_error = None
    interval_cache = None
    intervals = []
    neval = 0

    CONVERGED = 0
    NOT_CONVERGED = 1
    ROUNDING_ERROR = 2
    NOT_A_NUMBER = 3

    status_msg = {
        CONVERGED: "Target precision reached.",
        NOT_CONVERGED: "Target precision not reached.",
        ROUNDING_ERROR: "Target precision could not be reached due to rounding error.",
        NOT_A_NUMBER: "Non-finite values encountered."
    }

    with MapWrapper(workers) as mapwrapper:
        to_process = []

        for j in range(len(initial_intervals)):
            x1, x2 = initial_intervals[j]
            to_process.append((x1, x2, f, norm_func))

        for x1, x2, ig, err, rnd in mapwrapper(_quadrature, to_process):
            neval += _quadrature.num_eval

            if global_integral is None:
                if isinstance(ig, (float, complex)):
                    # Specialize for scalars
                    if norm_func in (_max_norm, np.linalg.norm):
                        norm_func = abs

                global_integral = ig
                global_error = float(err)
                rounding_error = float(rnd)

                cache_count = cache_size // _get_sizeof(ig)
                interval_cache = LRUDict(cache_count)
            else:
                global_integral += ig
                global_error += err
                rounding_error += rnd

            interval_cache[(x1, x2)] = copy.copy(ig)
            intervals.append((-err, x1, x2))

        heapq.heapify(intervals)
        ier = NOT_CONVERGED

        while intervals and len(intervals) < limit:
            # Select intervals with largest errors for subdivision
            tol = max(epsabs, epsrel*norm_func(global_integral))

            to_process = []
            err_sum = 0

            for j in range(parallel_count):
                if not intervals:
                    break

                if j > 0 and err_sum > global_error - tol/8:
                    # avoid unnecessary parallel splitting
                    break

                interval = heapq.heappop(intervals)

                neg_old_err, a, b = interval
                old_int = interval_cache.pop((a, b), None)
                to_process.append(((-neg_old_err, a, b, old_int), f, norm_func, _quadrature))
                err_sum += -neg_old_err

            # Subdivide intervals
            for dint, derr, dround_err, subint, dneval in mapwrapper(
                _subdivide_interval, to_process):
                neval += dneval
                global_integral += dint
                global_error += derr
                rounding_error += dround_err
                for x in subint:
                    x1, x2, ig, err = x
                    interval_cache[(x1, x2)] = ig
                    heapq.heappush(intervals, (-err, x1, x2))

            # Termination check
            if len(intervals) >= min_intervals:
                tol = max(epsabs, epsrel*norm_func(global_integral))
                if global_error < tol/8:
                    ier = CONVERGED
                    break
                if global_error < rounding_error:
                    ier = ROUNDING_ERROR
                    break

            if not (np.isfinite(global_error) and np.isfinite(rounding_error)):
                ier = NOT_A_NUMBER
                break

    res = global_integral
    err = global_error + rounding_error

    if full_output:
        res_arr = np.asarray(res)
        dummy = np.full(res_arr.shape, np.nan, dtype=res_arr.dtype)
        integrals = np.array([interval_cache.get((z[1], z[2]), dummy)
                                      for z in intervals], dtype=res_arr.dtype)
        errors = np.array([-z[0] for z in intervals])
        intervals = np.array([[z[1], z[2]] for z in intervals])

        info = _Bunch(neval=neval,
                      success=(ier == CONVERGED),
                      status=ier,
                      message=status_msg[ier],
                      intervals=intervals,
                      integrals=integrals,
                      errors=errors)
        return (res, err, info)
    else:
        return (res, err)
