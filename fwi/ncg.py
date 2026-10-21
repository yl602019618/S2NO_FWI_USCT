"""Original Poblano-style nonlinear conjugate-gradient optimizer."""

import numpy as np
import torch
from ._config import Bunch
import torch.nn as nn
import logging

# NCG   Nonlinear conjugate gradient minimization.


def ncg(NCG_param, FUN, x0):
    # Update Parameter
    param = POBLANO_PARAMS()
    param.update(NCG_param)
    # Initialize
    xk = x0
    fk, gk = FUN(xk)
    out = POBLANO_OUT(xk, fk, gk, 1, param)
    # Main loop
    while out.ExitFlag == -1:
        if out.Iters == 0:
            pk = -gk
            ak = 0.010
            gkTgk = torch.dot(torch.conj(gk.T.flatten()), gk.flatten())
            isIniIter = 1
        else:
            # Compute next direction
            if out.Iters % param.RestartIters == 0:
                # Restart to prevent stagnation
                bk = 0
                pk = -gk
            else:
                # Direction update
                if param.Update == "FR":
                    # Fletcher-Reeves
                    gkTgk = torch.dot(torch.conj(gk.T.flatten()), gk.flatten())
                    if gkTgkold > 0:
                        bk = gkTgk / gkTgkold
                    else:
                        logging.info(f"Warning: bk set to 0")
                        bk = 0
                elif param.Update == "PR":
                    # Polak-Ribiere
                    gkTgk = torch.dot(torch.conj(gk.T.flatten()), gk.flatten())
                    gkMgkold = gk - gkold
                    if gkTgkold > 0:
                        bk = (
                            torch.dot(torch.conj(gk.T.flatten()), gkMgkold.flatten())
                            / gkTgkold
                        )
                    else:
                        logging.info(f"Warning: bk set to 0")
                        bk = 0
                elif param.Update == "HS":
                    # Hestenes-Stiefel
                    gkMgkold = gk - gkold
                    denom = torch.dot(torch.conj(pkold.T.flatten()), gkMgkold.flatten())
                    if denom > 0:
                        bk = (
                            torch.dot(torch.conj(gk.T.flatten()), gkMgkold.flatten())
                            / denom
                        )
                    else:
                        logging.info(f"Warning: bk set to 0")
                        bk = 0
                elif param.Update == "SD":
                    # Steepest Descent
                    bk = 0
                else:
                    raise ValueError(
                        "Error: options.Update is not valid. Choices are {FR, PR, HS}"
                    )
                # Do not allow negative conjugate direction weights
                if bk < 0:
                    bk = max(0, bk)

                # Restart method from Nocedal and Wright
                if param.RestartNW:
                    v = param.RestartNWTol
                    if (
                        torch.dot(torch.conj(gk.T.flatten()), gkold.flatten())
                        / (gkTgkold**2)
                        >= v
                    ):
                        bk = 0

                # New direction
                pk = -gk + bk * pkold
        # xkold = xk
        # fkold = fk
        gkold = gk
        pkold = pk
        gkTgkold = gkTgk

        # Compute step length
        xk, fk, gk, ak, lsinfo, nfev, isIniIter = POBLANO_LINESEARCH(
            FUN, xk, fk, gk, ak, pk, param, isIniIter
        )
        if lsinfo != 1 and param.Display == "iter":
            logging.info(f"Warning: line search warning = {lsinfo}")

        # Update counts, check exit conditions, etc.
        out = POBLANO_OUT(xk, fk, gk, nfev, param, out)

    return out


def POBLANO_PARAMS():
    params = Bunch(
        {
            "RestartIters": 20,  # Number of iterations before restarting the conjugate direction
            "Update": "PR",  # Update method for conjugate direction: 'FR', 'PR', 'HS', or 'SD'
            "RestartNW": False,  # Flag to use restart heuristic of Nocedal and Wright
            "RestartNWTol": 0.1,  # Tolerance for Nocedal and Wright restart heuristic
            "Display": "iter",  # Controls the level of output information; can be 'iter', 'final', or 'off'
            "DisplayIters": 1,  # Frequency of iteration information output
            "MaxIters": 1000,  # Maximum number of iterations
            "MaxFuncEvals": 10000,  # Maximum number of function evaluations
            "StopTol": 1e-5,  # Tolerance for stopping criterion
            "RelFuncTol": 1e-6,  # Tolerance for relative change in function value
            "TraceX": False,  # Whether to record the variable values at each iteration
            "TraceFunc": False,  # Whether to record the function values at each iteration
            "TraceRelFunc": False,  # Whether to record the relative function values at each iteration
            "TraceGrad": False,  # Whether to record the gradient values at each iteration
            "TraceGradNorm": False,  # Whether to record the gradient norm at each iteration
            "TraceFuncEvals": False,  # Whether to record the function evaluations at each iteration
            "LineSearch_method": "more-thuente",  # Line search method, here it is 'more-thuente'
            "LineSearch_initialstep": 1,  # Initial step length for line search
            "LineSearch_xtol": 1e-15,  # xtol parameter for line search
            "LineSearch_ftol": 1e-4,  # ftol parameter for line search
            "LineSearch_gtol": 1e-2,  # gtol parameter for line search
            "LineSearch_stpmin": 1e-15,  # Minimum step length for line search
            "LineSearch_stpmax": 1e15,  # Maximum step length for line search
            "LineSearch_maxfev": 20,  # Maximum number of function evaluations in line search
            "LineSearch_ibound": False,  # Whether to use bound constraints
            "LineSearch_ub": float("inf"),  # Upper bound for line search
            "LineSearch_lb": -float("inf"),  # Lower bound for line search
        }
    )
    return params


def POBLANO_OUT(x, f, g, nfev, Params, out=None):
    """

     POBLANO_OUT   Standard output parameters for the Poblano Toolbox.

     Standard output parameters for Poblano are as follows:

     Args:
         x: Solution (i.e., best point found so far)
         f: Function value at solution
         g: Gradient at solution
         nfev: Number of function evaluations
         Params: Input parameters (dictionary)
         out: Previous output dictionary (if available)

     Returns:
         out: Updated output dictionary

    Standard ouptut parameters for Poblano are as follows:

      OUT.X        : solution (i.e., best point found so far)
      OUT.F        : function value at solution
      OUT.G        : gradient at solution
      OUT.Params   : input parameters
      OUT.FuncEvals: number of function evaluations
      OUT.Iters    : number of iterations
      OUT.ExitFlag : termination flag
       -1 = no termination condition has been statisfied
        0 = successful termination (based on StopTol)
        1 = maximum number of iterations exceeded
        2 = maximum number of function values exceeded
        3 = relative change in function value < RelFuncTol
        4 = NaNs found in f, g, or norm(g)

      OUT.ExitDescription : text description of termination flag

    Additional fields may be included if a trace is requested.
    """
    if out is None:
        out = Bunch(
            {
                "Params": Params,
                "ExitFlag": -1,
                "ExitDescription": "",
                "X": x,  # inverse vp
                "F": f,  # loss scale
                "G": g,  # gradient
                "FuncEvals": nfev,  # function evaluate number
                "Iters": 0,
                "steplen": (
                    Params["LineSearch_initialstep"]
                    if Params["LineSearch_initialstep"] > 0
                    else 1.0
                ),
            }
        )
        # Initialize traces if requested
        if Params.TraceX:
            out.TraceX = []
        if Params.TraceFunc:
            out.TraceFunc = []
        if Params.TraceRelFunc:
            out.TraceRelFunc = []
        if Params.TraceGrad:
            out.TraceGrad = []
        if Params.TraceGradNorm:
            out.TraceGradNorm = []
        if Params.TraceFuncEvals:
            out.TraceFuncEvals = []
    else:
        oldf = out.F
        if f <= oldf:
            out.X = x
            out.F = f
            out.G = g
        out.FuncEvals += nfev
        out.Iters += 1

        relfit = (
            abs(f - oldf)
            if abs(oldf) < torch.finfo(float).eps
            else abs((f - oldf) / oldf)
        )

    # Initialize variables
    nx = x.numel()
    # g2norm = np.linalg.norm(g) --> pytorch
    g2norm = torch.norm(g)
    g2normnx = g2norm / nx

    # Record trace info
    if Params.TraceX:
        out.TraceX.append(x)
    if Params.TraceFunc:
        out.TraceFunc.append(f)
    if Params.TraceFunc and Params.TraceRelFunc and out.Iters > 0:
        out.TraceRelFunc.append(relfit)
    if Params.TraceGrad:
        out.TraceGrad.append(g)
    if Params.TraceGradNorm:
        out.TraceGradNorm.append(g2norm)
    if Params.TraceFuncEvals:
        out.TraceFuncEvals.append(nfev)

    # Check termination conditions
    if g2normnx < Params.StopTol:
        out.ExitFlag = 0
        out.ExitDescription = "Successful termination based on StopTol"
    elif out.Iters >= Params.MaxIters:
        out.ExitFlag = 1
        out.ExitDescription = "Maximum number of iterations exceeded"
    elif out.FuncEvals >= Params.MaxFuncEvals:
        out.ExitFlag = 2
        out.ExitDescription = "Maximum number of function evaluations exceeded"
    elif out.Iters > 0 and relfit <= Params.RelFuncTol:
        out.ExitFlag = 3
        out.ExitDescription = "Relative change in F < RelFuncTol"
    elif torch.isnan(f) or torch.isnan(g).any() or torch.isnan(g2norm):
        out.ExitFlag = 4
        out.ExitDescription = "NaNs found in F, G, or norm(G)"

    # Display iteration information
    if out.Iters == 0 and Params.Display != "off":
        logging.info(
            " Iter  FuncEvals       F(X)          ||G(X)||/N       StepLength   "
        )
        logging.info(
            "------ --------- ---------------- ---------------- ----------------"
        )

    if Params.Display == "iter" and (
        out.Iters % Params.DisplayIters == 0 or out.ExitFlag >= 0
    ):
        logging.info(
            f"{out['Iters']:6d} {out['FuncEvals']:9d} {f:16.4e} {g2normnx:16.4e} {out['steplen']:16.4e}"
        )

    if out.ExitFlag >= 0 and Params.Display == "final":
        logging.info(
            f"{out['Iters']:6d} {out['FuncEvals']:9d} {out['F']:16.4e} {g2normnx:16.4e} {out['steplen']:16.4e}"
        )
    return out


def POBLANO_LINESEARCH(FUN, x0, f0, g0, a0, d0, Params, isIniIter):
    if Params.LineSearch_method == "more-thuente":
        minfun = cvsrch
    else:
        raise ValueError(f"Unknown LineSearch_method: {Params.LineSearch_method}")

    # Check whether user specified an initial step
    if Params.LineSearch_initialstep > 0 and isIniIter == 1:
        a0 = Params.LineSearch_initialstep
        isIniIter = 0
    xk, fk, gk, ak, lsinfo, nfev = minfun(FUN, x0, f0, g0, a0, d0, Params)
    return xk, fk, gk, ak, lsinfo, nfev, isIniIter


def cvsrch(fcn, x, f, g, stp, s, params):

    # Initialize params
    xtol = params.LineSearch_xtol
    ftol = params.LineSearch_ftol
    gtol = params.LineSearch_gtol
    stpmin = params.LineSearch_stpmin
    stpmax = params.LineSearch_stpmax
    maxfev = params.LineSearch_maxfev
    ibound = params.LineSearch_ibound
    ub = params.LineSearch_ub
    lb = params.LineSearch_lb

    # Initialize local variables
    # move up in case of early termination (before this was initialized
    nfev = 0

    p5 = 0.5
    p66 = 0.66
    xtrapf = 4.0
    info = 0
    infoc = 1

    n_x = x.numel()
    n_ub = ub.numel()
    n_lb = lb.numel()
    if (
        n_x <= 0
        or stp <= 0.0
        or ftol < 0.0
        or gtol < 0.0
        or xtol < 0.0
        or stpmin < 0.0
        or stpmax < stpmin
        or maxfev <= 0
    ):
        return x, f, g, stp, info, nfev

    if ibound and (n_ub != n_x or n_lb != n_x):
        logging.info(f"improper set of limits:")
        logging.info(f"size of x is {n_x}")
        logging.info(f"size of ub is {n_ub}")
        logging.info(f"size of lb is {n_lb}")

        return x, f, g, stp, info, nfev

    dginit = torch.dot(torch.conj(g.T.flatten()), torch.conj(s.flatten()))
    if dginit >= 0.0:
        return x, f, g, stp, info, nfev

    # More initializations
    brackt = False
    stage1 = True
    nfev = 0
    finit = f
    dgtest = ftol * dginit
    width = stpmax - stpmin
    width1 = 2 * width
    wa = x.clone()

    stx = 0.0
    fx = finit
    dgx = dginit
    sty = 0.0
    fy = finit
    dgy = dginit
    while True:  # begin to iteration
        if brackt:
            stmin = min(stx, sty)
            stmax = max(stx, sty)
        else:
            stmin = stx
            stmax = stp + xtrapf * (stp - stx)

        stp = max(stp, stpmin)
        stp = min(stp, stpmax)

        if (
            (brackt and (stp <= stmin or stp >= stmax))
            or nfev >= maxfev
            or infoc == 0
            or (brackt and stmax - stmin <= xtol * stmax)
        ):
            stp = stx

        x = wa + stp * s
        if ibound:
            x = torch.minimum(x, ub)
            x = torch.maximum(x, lb)

        f, g = fcn(x)
        nfev += 1
        dg = torch.dot(torch.conj(g.T.flatten()), torch.conj(s.flatten()))
        ftest1 = finit + stp * dgtest

        if (brackt and (stp <= stmin or stp >= stmax)) or infoc == 0:
            info = 6
        if stp == stpmax and f <= ftest1 and dg <= dgtest:
            info = 5
        if stp == stpmin and (f > ftest1 or dg >= dgtest):
            info = 4
        if nfev >= maxfev:
            info = 3
        if brackt and stmax - stmin <= xtol * stmax:
            info = 2
        if f <= ftest1 and abs(dg) <= gtol * -dginit:
            info = 1

        if info != 0:
            return x, f, g, stp, info, nfev

        if stage1 and f <= ftest1 and dg >= min(ftol, gtol) * dginit:
            stage1 = False

        if stage1 and f <= fx and f > ftest1:
            fm = f - stp * dgtest
            fxm = fx - stx * dgtest
            fym = fy - sty * dgtest
            dgm = dg - dgtest
            dgxm = dgx - dgtest
            dgym = dgy - dgtest

            stx, fxm, dgxm, sty, fym, dgym, stp, fm, dgm, brackt, infoc = cstep(
                stx, fxm, dgxm, sty, fym, dgym, stp, fm, dgm, brackt, stmin, stmax
            )

            fx = fxm + stx * dgtest
            fy = fym + sty * dgtest
            dgx = dgxm + dgtest
            dgy = dgym + dgtest
        else:
            stx, fx, dgx, sty, fy, dgy, stp, f, dg, brackt, infoc = cstep(
                stx, fx, dgx, sty, fy, dgy, stp, f, dg, brackt, stmin, stmax
            )

        if brackt:
            if abs(sty - stx) >= p66 * width1:
                stp = stx + p5 * (sty - stx)
            width1 = width
            width = abs(sty - stx)


def cstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    p66 = 0.66
    info = torch.tensor(0)
    # Check input parameters for errors
    if (
        brackt & ((stp <= min(stx, sty)) | (stp >= max(stx, sty)))
        | (dx * (stp - stx) >= 0.0)
        | (stpmax < stpmin)
    ):
        return stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, info

    # Determine if the derivatives have opposite sign
    sgnd = dp * (dx / torch.abs(dx))

    # First case: A higher function value
    if fp > fx:
        info = torch.tensor(1)
        bound = torch.tensor(1)
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = torch.norm(torch.tensor([theta, dx, dp]), float("inf"))
        gamma = s * torch.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp < stx:
            gamma = -gamma
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2) * (stp - stx)
        if torch.abs(stpc - stx) < torch.abs(stpq - stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc) / 2
        brackt = torch.tensor(1)

    # Second case: A lower function value and derivatives of opposite sign
    elif sgnd < 0.0:
        info = torch.tensor(2)
        bound = torch.tensor(0)
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = torch.norm(torch.tensor([theta, dx, dp]), float("inf"))
        gamma = s * torch.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if torch.abs(stpc - stp) > torch.abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = torch.tensor(1)

    # Third case: A lower function value, derivatives of the same sign, and the magnitude of the derivative decreases
    elif torch.abs(dp) < torch.abs(dx):
        info = torch.tensor(3)
        bound = torch.tensor(1)
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = torch.norm(torch.tensor([theta, dx, dp]), float("inf"))
        gamma = s * torch.sqrt(
            torch.max(torch.tensor(0.0), (theta / s) ** 2 - (dx / s) * (dp / s))
        )
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q
        if (r < 0.0) & (gamma != 0.0):
            stpc = stp + r * (stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if brackt:
            if abs(stp - stpc) < abs(stp - stpq):
                stpf = stpc
            else:
                stpf = stpq
        else:
            if abs(stp - stpc) > abs(stp - stpq):
                stpf = stpc
            else:
                stpf = stpq

    # Fourth case: A lower function value, derivatives of the same sign, and the magnitude of the derivative does not decrease
    else:
        info = torch.tensor(4)
        bound = torch.tensor(0)
        if brackt:
            theta = 3 * (fp - fy) / (sty - stp) + dy + dp
            s = torch.norm(torch.tensor([theta, dy, dp]), float("inf"))
            gamma = s * torch.sqrt((theta / s) ** 2 - (dy / s) * (dp / s))
            if stp > sty:
                gamma = -gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p / q
            stpc = stp + r * (sty - stp)
            stpf = stpc
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin

    # Update the interval of uncertainty
    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < 0.0:
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp

    # Compute the new step and safeguard it
    stpf = min(stpmax, stpf)
    stpf = max(stpmin, stpf)
    stp = stpf
    if brackt & bound:
        if sty > stx:
            stp = min(stx + p66 * (sty - stx), stp)
        else:
            stp = max(stx + p66 * (sty - stx), stp)

    return stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, info
