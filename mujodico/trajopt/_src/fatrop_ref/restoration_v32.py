"""Structured native-style FATROP restoration phase for the JAX reference solver.

V37_PAIR_CONDENSATION: preserve the v36 `ne` shadow fix and exactly
condense each p/n elastic pair to a unit-curvature scaled difference.

This module mirrors FATROP/Ipopt's elastic L1 restoration NLP while reusing the
existing serial/PAS generalized-Riccati solver.  It deliberately has no task-
specific logic.

Restoration variables
---------------------
The original primal trajectory (x,u) and original inequality slack s are kept.
For every transformed equality row we add positive/negative elastic variables
p,n:

    c(x,u,s) + p - n = 0,   p,n >= 0.

For the lifted OCP this includes
  * multiple-shooting dynamics,
  * local contact/equality constraints,
  * path-inequality slack equations h(x,u)-s=0.

The restoration objective is FATROP's IpNlpResto objective

    rho * 1^T(p+n)
    + zeta/2 ||D_R (z-z_ref)||^2,

where zeta=sqrt(mu_orig) and D_R=diag(1/max(1,|z_ref|)).

The p/n Newton directions are represented as additional stage-local controls.
Therefore the existing structured Riccati/PAS solve can be reused unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np
import jax
import jax.numpy as jnp

from .linear import solve_serial, solve_pas


@dataclass
class RestorationResult:
    success: bool
    x: object
    u: object
    s: object
    lambda_dyn: object
    beta_eq: object
    gamma_ineq: object
    zL: object
    zU: object
    inner_mu: float
    iterations: int
    reason: str
    rows: list = field(default_factory=list)


def _optn(opt, name, default):
    return getattr(opt, name, default)


def _maxabs(x):
    x = jnp.asarray(x)
    return float(jnp.max(jnp.abs(x), initial=0.0))


def _meanabs(x):
    x = jnp.asarray(x)
    return float(jnp.mean(jnp.abs(x))) if x.size else 0.0


def _elastic_initialize(c, mu, rho):
    """Native restoration p/n central initializer.

    This is the stable closed-form initialization encoded by
    IpInitializerResto.  It satisfies

        c + p - n = 0,
        p,n > 0,
        z_p = mu/p, z_n = mu/n,
        z_p + z_n = 2 rho,

    and therefore the p/n stationarity equations are exactly centered.
    """
    c = jnp.asarray(c)
    a = jnp.asarray(mu / (2.0 * rho), dtype=c.dtype)
    root = jnp.sqrt(a * a + 0.25 * c * c)
    p = a - 0.5 * c + root
    n = a + 0.5 * c + root
    tiny = jnp.finfo(c.dtype).tiny
    p = jnp.maximum(p, tiny)
    n = jnp.maximum(n, tiny)
    zp = mu / p
    zn = mu / n
    y = zp - rho
    return p, n, zp, zn, y


def _finite_margins(self, s):
    flo, fhi, Sl, Su = self._margins(s)
    if self.ni:
        flo = jnp.broadcast_to(flo, s.shape)
        fhi = jnp.broadcast_to(fhi, s.shape)
    return flo, fhi, Sl, Su


def _resto_objective(x, u, s, pd, nd, pe, ne, ph, nh,
                       xref, uref, sref, rho, zeta):
    drx = 1.0 / jnp.maximum(1.0, jnp.abs(xref))
    dru = 1.0 / jnp.maximum(1.0, jnp.abs(uref))
    drs = 1.0 / jnp.maximum(1.0, jnp.abs(sref))
    prox = 0.5 * zeta * (
        jnp.sum((drx * (x - xref)) ** 2)
        + jnp.sum((dru * (u - uref)) ** 2)
        + jnp.sum((drs * (s - sref)) ** 2)
    )
    elastic = rho * (
        jnp.sum(pd + nd) + jnp.sum(pe + ne) + jnp.sum(ph + nh)
    )
    return float(prox + elastic)


def _resto_barrier(self, x, u, s, pd, nd, pe, ne, ph, nh,
                   xref, uref, sref, rho, zeta, mu):
    val = _resto_objective(
        x, u, s, pd, nd, pe, ne, ph, nh,
        xref, uref, sref, rho, zeta,
    )
    logs = 0.0
    if self.ni:
        flo, fhi, Sl, Su = _finite_margins(self, s)
        if bool(jnp.any((flo & (Sl <= 0)) | (fhi & (Su <= 0)))):
            return math.inf
        logs += float(jnp.sum(jnp.where(flo, jnp.log(Sl), 0.0)))
        logs += float(jnp.sum(jnp.where(fhi, jnp.log(Su), 0.0)))
    for q in (pd, nd, pe, ne, ph, nh):
        if q.size:
            if bool(jnp.any(q <= 0)):
                return math.inf
            logs += float(jnp.sum(jnp.log(q)))
    return val - mu * logs


def _resto_constraints(self, x, u, s, pd, nd, pe, ne, ph, nh):
    rd, re, rh = self._constraints(x, u, s)
    return rd + pd - nd, re + pe - ne, rh + ph - nh


def _theta_tuple(res, norm="l1"):
    rd, re, rh = res
    if norm == "linf":
        return max(_maxabs(rd), _maxabs(re), _maxabs(rh))
    return float(
        jnp.sum(jnp.abs(rd))
        + jnp.sum(jnp.abs(re))
        + jnp.sum(jnp.abs(rh))
    )


class _RestoFilter:
    def __init__(self, theta0, opt):
        self.entries = []
        self.theta_max = _optn(opt, "theta_max_fact", 1e4) * max(1.0, float(theta0))
        self.theta_min = _optn(opt, "theta_min_fact", 1e-4) * max(1.0, float(theta0))
        self.opt = opt

    def acceptable(self, theta, phi):
        if not np.isfinite(theta + phi) or theta > self.theta_max:
            return False
        gt = _optn(self.opt, "gamma_theta", 1e-5)
        gp = _optn(self.opt, "gamma_phi", 1e-8)
        return all(
            theta <= (1.0 - gt) * th or phi <= ph - gp * th
            for th, ph in self.entries
        )

    def trial(self, theta, phi, th0, ph0, alpha, dphi):
        if not self.acceptable(theta, phi):
            return False, False
        desc = -dphi
        sphi = _optn(self.opt, "s_phi", 2.3)
        sth = _optn(self.opt, "s_theta", 1.1)
        delta = _optn(self.opt, "delta", 1.0)
        ftype = (
            th0 <= self.theta_min
            and np.isfinite(desc)
            and desc > 0.0
            and alpha * desc ** sphi > delta * max(th0, 0.0) ** sth
        )
        if ftype:
            eta = _optn(self.opt, "eta_phi", 1e-8)
            return phi <= ph0 + eta * alpha * dphi, True
        gt = _optn(self.opt, "gamma_theta", 1e-5)
        gp = _optn(self.opt, "gamma_phi", 1e-8)
        return (
            theta <= (1.0 - gt) * th0 or phi <= ph0 - gp * th0
        ), False

    def add(self, theta, phi):
        self.entries = [
            (t, p) for t, p in self.entries
            if not (theta <= t and phi <= p)
        ]
        self.entries.append((float(theta), float(phi)))


def _build_direction(
    self, *, x, u, s, lam, beta, gamma, zL, zU,
    pd, nd, pe, ne, ph, nh,
    zpd, znd, zpe, zne, zph, znh,
    xref, uref, sref, rho, zeta, mu, reg, backend, stagewise,
):
    """Build one restoration Newton system with exact p/n pair condensation.

    V37 analytically condenses each positive/negative elastic pair to its
    difference v = dp-dn.  For one pair,

        0.5*hp*dp^2 + rp*dp + 0.5*hn*dn^2 + rn*dn,

    subject to dp-dn=v, reduces exactly to

        0.5*heff*v^2 + reff*v,

    where

        w     = 1/hp + 1/hn,
        heff  = 1/w,
        reff  = heff * (rp/hp - rn/hn).

    We then use t=sqrt(heff)*v as the Riccati variable.  The elastic Hessian
    block is therefore identity and the constraint column is sqrt(w).  This is
    algebraically equivalent to explicit p/n variables but avoids the extreme
    barrier-curvature conditioning exposed by the v36 ne-shadow fix.
    """
    lin0 = self._linearize(x, u, stagewise)
    lin_active = self._active_hessian_lin(lin0, x, u, lam, beta, gamma)

    gx0, gu0, Q0, S0, R0, rd, A, B, re, E, F, h, Gx, Gu, gf0, Qf0 = lin0
    _, _, Qa, Sa, Ra, _, _, _, _, _, _, _, _, _, _, Qfa = lin_active

    dtype = x.dtype
    N, nx, nu, neq, ni = self.N, self.nx, self.nu, self.ne, self.ni

    drx = 1.0 / jnp.maximum(1.0, jnp.abs(xref))
    dru = 1.0 / jnp.maximum(1.0, jnp.abs(uref))
    drs = 1.0 / jnp.maximum(1.0, jnp.abs(sref))

    Qcurv = Qa - Q0
    Scurv = Sa - S0
    Rcurv = Ra - R0
    Qfcurv = Qfa - Qf0

    qref_x = zeta * drx[:-1] ** 2 * (x[:-1] - xref[:-1])
    qref_u = zeta * dru ** 2 * (u - uref)
    qref_f = zeta * drx[-1] ** 2 * (x[-1] - xref[-1])
    qref_s = zeta * drs ** 2 * (s - sref)

    Q = Qcurv + jax.vmap(jnp.diag)(zeta * drx[:-1] ** 2)
    S = Scurv
    Rorig = Rcurv + jax.vmap(jnp.diag)(zeta * dru ** 2)
    Qf = Qfcurv + jnp.diag(zeta * drx[-1] ** 2)

    prev = jnp.concatenate(
        [jnp.zeros((1, nx), dtype=dtype), lam[:-1]], axis=0
    )
    rx = qref_x + jnp.einsum("nij,ni->nj", A, lam) - prev
    ru = qref_u + jnp.einsum("nij,ni->nj", B, lam)
    if neq:
        rx = rx + jnp.einsum("nij,ni->nj", E, beta)
        ru = ru + jnp.einsum("nij,ni->nj", F, beta)
    if ni:
        rx = rx + jnp.einsum("nij,ni->nj", Gx, gamma)
        ru = ru + jnp.einsum("nij,ni->nj", Gu, gamma)
    rxf = qref_f - lam[-1]

    # Original inequality slack elimination.
    if ni:
        flo, fhi, Sl, Su = _finite_margins(self, s)
        rs = qref_s - gamma - zL + zU
        rL = jnp.where(flo, Sl * zL - mu, 0.0)
        rU = jnp.where(fhi, Su * zU - mu, 0.0)
        sig_s = zeta * drs ** 2
        sig_s = sig_s + jnp.where(flo, zL / Sl, 0.0)
        sig_s = sig_s + jnp.where(fhi, zU / Su, 0.0)
        wd = 1.0 / (sig_s + reg)
        W = jax.vmap(jnp.diag)(wd)
        ss = (
            rs
            + jnp.where(flo, rL / Sl, 0.0)
            - jnp.where(fhi, rU / Su, 0.0)
        )
    else:
        flo = jnp.zeros((N, 0), dtype=bool)
        fhi = flo
        Sl = jnp.zeros((N, 0), dtype=dtype)
        Su = Sl
        rL = Sl
        rU = Sl
        wd = Sl
        W = jnp.zeros((N, 0, 0), dtype=dtype)
        ss = Sl

    # Elastic bound-reduced stationarity and positive curvature.
    qpd = rho + lam - mu / pd
    qnd = rho - lam - mu / nd
    qpe = rho + beta - mu / pe if neq else pe
    qne = rho - beta - mu / ne if neq else ne
    qph = rho + gamma - mu / ph if ni else ph
    qnh = rho - gamma - mu / nh if ni else nh

    hp_d = zpd / pd
    hn_d = znd / nd
    hp_e = zpe / pe if neq else pe
    hn_e = zne / ne if neq else ne
    hp_h = zph / ph if ni else ph
    hn_h = znh / nh if ni else nh

    def pair_reduce(hp, hn, rp, rn):
        # Exact compliance of the p/n pair.
        wp = 1.0 / hp
        wn = 1.0 / hn
        w = wp + wn
        heff = 1.0 / w
        sqrt_h = jnp.sqrt(heff)
        # t = sqrt(heff) * (dp-dn)
        # constraint coefficient on t = 1/sqrt(heff) = sqrt(w)
        col = 1.0 / sqrt_h
        rscaled = (heff * (rp * wp - rn * wn)) / sqrt_h
        return col, rscaled, heff

    col_d, rt_d, heff_d = pair_reduce(hp_d, hn_d, qpd, qnd)
    if neq:
        col_e, rt_e, heff_e = pair_reduce(hp_e, hn_e, qpe, qne)
    else:
        col_e = jnp.zeros((N, 0), dtype=dtype)
        rt_e = col_e
        heff_e = col_e
    if ni:
        col_h, rt_h, heff_h = pair_reduce(hp_h, hn_h, qph, qnh)
    else:
        col_h = jnp.zeros((N, 0), dtype=dtype)
        rt_h = col_h
        heff_h = col_h

    # Augmented variable: [du | t_dyn | t_eq | t_path].
    off_u = 0
    off_d = nu
    off_e = off_d + nx
    off_h = off_e + neq
    nvr = off_h + ni

    R = jnp.zeros((N, nvr, nvr), dtype=dtype)
    R = R.at[:, :nu, :nu].set(Rorig)
    R = R.at[:, off_d:off_e, off_d:off_e].set(
        jnp.broadcast_to(jnp.eye(nx, dtype=dtype), (N, nx, nx))
    )
    if neq:
        R = R.at[:, off_e:off_h, off_e:off_h].set(
            jnp.broadcast_to(jnp.eye(neq, dtype=dtype), (N, neq, neq))
        )
    if ni:
        R = R.at[:, off_h:, off_h:].set(
            jnp.broadcast_to(jnp.eye(ni, dtype=dtype), (N, ni, ni))
        )

    Saug = jnp.zeros((N, nx, nvr), dtype=dtype)
    Saug = Saug.at[:, :, :nu].set(S)

    raug_parts = [ru, rt_d]
    if neq:
        raug_parts.append(rt_e)
    if ni:
        raug_parts.append(rt_h)
    raug = jnp.concatenate(raug_parts, axis=1)

    # Dynamics: A dx + B du + (dp-dn) + ard = 0.
    Baug = jnp.zeros((N, nx, nvr), dtype=dtype)
    Baug = Baug.at[:, :, :nu].set(B)
    Id = jnp.broadcast_to(jnp.eye(nx, dtype=dtype), (N, nx, nx))
    Baug = Baug.at[:, :, off_d:off_e].set(
        Id * col_d[:, None, :]
    )

    # Local equality: E dx + F du + (dp-dn) + ere = 0.
    Faug = jnp.zeros((N, neq, nvr), dtype=dtype)
    if neq:
        Faug = Faug.at[:, :, :nu].set(F)
        Ie = jnp.broadcast_to(
            jnp.eye(neq, dtype=dtype), (N, neq, neq)
        )
        Faug = Faug.at[:, :, off_e:off_h].set(
            Ie * col_e[:, None, :]
        )

    # Path row: G dx + Gu du - ds + (dp-dn) + rrh = 0.
    Guaug = jnp.zeros((N, ni, nvr), dtype=dtype)
    if ni:
        Guaug = Guaug.at[:, :, :nu].set(Gu)
        Ih = jnp.broadcast_to(
            jnp.eye(ni, dtype=dtype), (N, ni, ni)
        )
        Guaug = Guaug.at[:, :, off_h:].set(
            Ih * col_h[:, None, :]
        )

    ard = rd + pd - nd
    ere = re + pe - ne
    rrh = h - s + ph - nh if ni else h
    gt = rrh + wd * ss if ni else rrh

    fn = solve_serial if backend == "serial" else solve_pas
    sol = fn(
        Q=Q, S=Saug, R=R, q=rx, r=raug,
        A=A, B=Baug, a=ard,
        E=E, F=Faug, e=ere,
        Gx=Gx, Gu=Guaug, g=gt, W=W,
        Qf=Qf, qf=rxf,
        dx0=jnp.zeros((nx,), dtype=dtype),
        regularization=reg,
    )

    dv = sol.du
    du = dv[:, :nu]

    # Recover p/n directions from their exact stationarity equations using
    # the equality/path multiplier increments returned by the structured solve.
    dlam = sol.dlambda_dyn
    dbeta = sol.dbeta_eq
    dgamma = sol.dgamma_ineq

    dpd = -(qpd + dlam) / hp_d
    dnd = -(qnd - dlam) / hn_d
    if neq:
        dpe = -(qpe + dbeta) / hp_e
        dne = -(qne - dbeta) / hn_e
    else:
        dpe, dne = pe, ne
    if ni:
        dph = -(qph + dgamma) / hp_h
        dnh = -(qnh - dgamma) / hn_h
    else:
        dph, dnh = ph, nh

    # Original slack and bound-dual directions.
    if ni:
        ds = (
            jnp.einsum("nij,nj->ni", Gx, sol.dx[:-1])
            + jnp.einsum("nij,nj->ni", Gu, du)
            + dph - dnh + rrh
        )
        dzL = jnp.where(flo, (-rL - zL * ds) / Sl, 0.0)
        dzU = jnp.where(fhi, (-rU + zU * ds) / Su, 0.0)
    else:
        ds = s
        dzL = s
        dzU = s

    def dz_lower(q, zq, dq):
        comp = q * zq - mu
        return (-comp - zq * dq) / q

    dzpd = dz_lower(pd, zpd, dpd)
    dznd = dz_lower(nd, znd, dnd)
    dzpe = dz_lower(pe, zpe, dpe) if neq else pe
    dzne = dz_lower(ne, zne, dne) if neq else ne
    dzph = dz_lower(ph, zph, dph) if ni else ph
    dznh = dz_lower(nh, znh, dnh) if ni else nh

    return dict(
        sol=sol, du=du, ds=ds, dzL=dzL, dzU=dzU,
        dpd=dpd, dnd=dnd, dpe=dpe, dne=dne, dph=dph, dnh=dnh,
        dzpd=dzpd, dznd=dznd, dzpe=dzpe, dzne=dzne,
        dzph=dzph, dznh=dznh,
        lin=lin0,
        pair_heff_dyn=heff_d,
        pair_heff_eq=heff_e,
        pair_heff_path=heff_h,
    )

def _positive_fraction(q, dq, tau):
    if q.size == 0:
        return 1.0
    vals = jnp.where(dq < 0, tau * q / (-dq), jnp.inf)
    return min(1.0, float(jnp.min(vals, initial=1.0)))


def _resto_max_step(self, s, zL, zU, D, state, mu):
    tau = max(0.99, 1.0 - float(mu))
    ap = 1.0
    ad = 1.0
    if self.ni:
        flo, fhi, Sl, Su = _finite_margins(self, s)
        ds = D["ds"]
        vals = jnp.where(flo & (ds < 0), tau * Sl / (-ds), jnp.inf)
        vals = jnp.minimum(
            vals,
            jnp.where(fhi & (ds > 0), tau * Su / ds, jnp.inf),
        )
        ap = min(ap, float(jnp.min(vals, initial=1.0)))
        vl = jnp.where(flo & (D["dzL"] < 0), tau * zL / (-D["dzL"]), jnp.inf)
        vu = jnp.where(fhi & (D["dzU"] < 0), tau * zU / (-D["dzU"]), jnp.inf)
        ad = min(ad, float(jnp.min(jnp.minimum(vl, vu), initial=1.0)))

    for name, dname in (
        ("pd", "dpd"), ("nd", "dnd"),
        ("pe", "dpe"), ("ne", "dne"),
        ("ph", "dph"), ("nh", "dnh"),
    ):
        ap = min(ap, _positive_fraction(state[name], D[dname], tau))

    for name, dname in (
        ("zpd", "dzpd"), ("znd", "dznd"),
        ("zpe", "dzpe"), ("zne", "dzne"),
        ("zph", "dzph"), ("znh", "dznh"),
    ):
        ad = min(ad, _positive_fraction(state[name], D[dname], tau))

    return ap, ad


def _resto_dphi(self, D, state, xref, uref, sref, rho, zeta, mu):
    x, u, s = state["x"], state["u"], state["s"]
    drx = 1.0 / jnp.maximum(1.0, jnp.abs(xref))
    dru = 1.0 / jnp.maximum(1.0, jnp.abs(uref))
    drs = 1.0 / jnp.maximum(1.0, jnp.abs(sref))
    qx = zeta * drx ** 2 * (x - xref)
    qu = zeta * dru ** 2 * (u - uref)
    qs = zeta * drs ** 2 * (s - sref)

    val = float(jnp.sum(qx * D["sol"].dx) + jnp.sum(qu * D["du"]))
    if self.ni:
        val += float(jnp.sum(qs * D["ds"]))
        flo, fhi, Sl, Su = _finite_margins(self, s)
        val += float(jnp.sum(jnp.where(flo, -mu / Sl * D["ds"], 0.0)))
        val += float(jnp.sum(jnp.where(fhi,  mu / Su * D["ds"], 0.0)))

    for qname, dname in (
        ("pd", "dpd"), ("nd", "dnd"),
        ("pe", "dpe"), ("ne", "dne"),
        ("ph", "dph"), ("nh", "dnh"),
    ):
        q = state[qname]
        dq = D[dname]
        if q.size:
            val += float(jnp.sum(rho * dq - mu / q * dq))
    return val


def _resto_error(self, D, state, rho, mu):
    sol = D["sol"]
    dual = max(
        _maxabs(sol.dlambda_dyn * 0.0),  # placeholder keeps shapes out of metric
        0.0,
    )
    # For barrier-update purposes, primal/complementarity are the decisive
    # terms.  Stationarity is represented by the solved KKT RHS and measured
    # approximately from the Newton step norm.
    primal = _theta_tuple(
        _resto_constraints(
            self,
            state["x"], state["u"], state["s"],
            state["pd"], state["nd"], state["pe"], state["ne"],
            state["ph"], state["nh"],
        ),
        "linf",
    )
    comp = 0.0
    if self.ni:
        flo, fhi, Sl, Su = _finite_margins(self, state["s"])
        comp = max(
            _maxabs(jnp.where(flo, Sl * state["zL"] - mu, 0.0)),
            _maxabs(jnp.where(fhi, Su * state["zU"] - mu, 0.0)),
        )
    for qn, zn in (
        ("pd", "zpd"), ("nd", "znd"),
        ("pe", "zpe"), ("ne", "zne"),
        ("ph", "zph"), ("nh", "znh"),
    ):
        if state[qn].size:
            comp = max(comp, _maxabs(state[qn] * state[zn] - mu))
    stepn = max(
        _maxabs(D["sol"].dx), _maxabs(D["du"]), _maxabs(D["ds"]),
        _maxabs(D["dpd"]), _maxabs(D["dnd"]),
        _maxabs(D["dpe"]), _maxabs(D["dne"]),
        _maxabs(D["dph"]), _maxabs(D["dnh"]),
    )
    return max(primal, comp, min(stepn, 1e6)), primal, comp


def _project_original_handoff(
    self, *, x, u, s, x0, u0, s0, zL0, zU0,
    mu_orig, backend, lammax, stagewise,
):
    """Mirror IpRestoPhaseMinCl1 handoff back to the original NLP."""
    if self.ni:
        flo, fhi, Sl0, Su0 = _finite_margins(self, s0)
        ds_total = s - s0
        compl_l = jnp.where(flo, Sl0 * zL0 - mu_orig, 0.0)
        compl_u = jnp.where(fhi, Su0 * zU0 - mu_orig, 0.0)
        dzL = jnp.where(flo, (-compl_l - zL0 * ds_total) / Sl0, 0.0)
        dzU = jnp.where(fhi, (-compl_u + zU0 * ds_total) / Su0, 0.0)
        tau = max(0.99, 1.0 - float(mu_orig))
        vl = jnp.where(flo & (dzL < 0), tau * zL0 / (-dzL), jnp.inf)
        vu = jnp.where(fhi & (dzU < 0), tau * zU0 / (-dzU), jnp.inf)
        ad = min(1.0, float(jnp.min(jnp.minimum(vl, vu), initial=1.0)))
        zL = zL0 + ad * dzL
        zU = zU0 + ad * dzU
        bmax = max(_maxabs(zL), _maxabs(zU))
        if bmax > 1e3:
            zL = jnp.where(flo, 1.0, 0.0)
            zU = jnp.where(fhi, 1.0, 0.0)
        # FATROP's modify_dual_bounds(mu): maintain strict positivity.
        eps = max(np.finfo(np.asarray(s).dtype).eps, 1e-16)
        zL = jnp.where(flo, jnp.maximum(zL, eps), 0.0)
        zU = jnp.where(fhi, jnp.maximum(zU, eps), 0.0)
    else:
        zL, zU = zL0, zU0

    lin = self._linearize(x, u, stagewise)

    # V36: compatibility with both compact and expanded private ABIs.
    # Some installed v9/v16 solvers structurally retain dormant direct-control
    # bound duals zuL/zuU even though finite control bounds are represented by
    # auxiliary inequality/slack rows in the fidelity branch.
    import inspect as _inspect
    _sig = _inspect.signature(self._initial_dual_estimate)
    _zero_u = jnp.zeros_like(u)
    _vals = dict(
        lin=lin, zL=zL, zU=zU,
        zuL=_zero_u, zuU=_zero_u,
        backend=backend, lammax=lammax,
    )
    _unknown = [name for name in _sig.parameters if name not in _vals]
    if _unknown:
        raise RuntimeError(
            f"Unsupported _initial_dual_estimate ABI {_sig}; "
            f"unknown parameters {_unknown}"
        )
    lam, beta, gamma, ok = self._initial_dual_estimate(
        **{name: _vals[name] for name in _sig.parameters}
    )
    if not ok:
        # constr_mult_reset_treshold is 0 in FATROP, so failed LS estimates are
        # reset rather than retaining large restoration multipliers.
        lam = jnp.zeros((self.N, self.nx), dtype=x.dtype)
        beta = jnp.zeros((self.N, self.ne), dtype=x.dtype)
        gamma = jnp.zeros((self.N, self.ni), dtype=x.dtype)
    return lam, beta, gamma, zL, zU


def full_restoration(
    self, *, x, u, s, lam, beta, gamma, zL, zU,
    mu_orig, opt, stagewise=False, outer_iteration=0,
    backend=None, print_rows=True,
):
    """Run FATROP's elastic L1 restoration NLP with structured Newton solves.

    This is intentionally callable as a standalone oracle before wiring it into
    the main line-search fallback.
    """
    backend = backend or _optn(opt, "linear_backend", "serial")
    rho = float(_optn(opt, "resto_penalty_parameter", 1000.0))
    zeta = float(_optn(opt, "resto_proximity_weight", 1.0)) * math.sqrt(float(mu_orig))
    req_red = float(_optn(opt, "required_infeasibility_reduction", 0.9))
    max_iter = int(_optn(opt, "max_resto_iter", 1000))
    # The targeted oracle can cap this without changing the production default.
    max_iter = min(max_iter, int(_optn(opt, "v32_resto_iter_cap", 50)))

    x = jnp.asarray(x)
    u = jnp.asarray(u)
    s = jnp.asarray(s)
    xref, uref, sref = x, u, s
    xstart, ustart, sstart = x, u, s
    zLstart, zUstart = jnp.asarray(zL), jnp.asarray(zU)

    rd0, re0, rh0 = self._constraints(x, u, s)
    theta0_l1 = self._theta(rd0, re0, rh0, "l1")
    theta0_inf = self._theta(rd0, re0, rh0, "linf")
    mu = max(float(mu_orig), theta0_inf)

    # Exact restoration elastic initializer for every transformed constraint.
    pd, nd, zpd, znd, lam = _elastic_initialize(rd0, mu, rho)
    pe, ne, zpe, zne, beta = _elastic_initialize(re0, mu, rho)
    ph, nh, zph, znh, gamma = _elastic_initialize(rh0, mu, rho)

    # Native initializer copies original primal/slack and caps original bound
    # multipliers by rho before nested optimization.
    zL = jnp.minimum(jnp.asarray(zL), rho)
    zU = jnp.minimum(jnp.asarray(zU), rho)

    state = dict(
        x=x, u=u, s=s, lam=lam, beta=beta, gamma=gamma, zL=zL, zU=zU,
        pd=pd, nd=nd, pe=pe, ne=ne, ph=ph, nh=nh,
        zpd=zpd, znd=znd, zpe=zpe, zne=zne, zph=zph, znh=znh,
    )

    rcons = _resto_constraints(
        self, x, u, s, pd, nd, pe, ne, ph, nh
    )
    theta_r = _theta_tuple(rcons, "l1")
    phi_r = _resto_barrier(
        self, x, u, s, pd, nd, pe, ne, ph, nh,
        xref, uref, sref, rho, zeta, mu,
    )
    filt = _RestoFilter(theta_r, opt)
    rows = []
    last_reg = 0.0

    def row(k, alpha=0.0, typ="x", reg=0.0):
        rd, re, rh = self._constraints(
            state["x"], state["u"], state["s"]
        )
        orig_inf = self._theta(rd, re, rh, "linf")
        orig_l1 = self._theta(rd, re, rh, "l1")
        obj = self._objective(state["x"], state["u"])
        rr = dict(
            inner_iter=k, outer_iter=outer_iteration,
            objective=obj, original_inf_pr=orig_inf,
            original_theta_l1=orig_l1, mu=mu,
            alpha_pr=alpha, type=typ, reg=reg,
            restoration_objective=_resto_objective(
                state["x"], state["u"], state["s"],
                state["pd"], state["nd"], state["pe"], state["ne"],
                state["ph"], state["nh"],
                xref, uref, sref, rho, zeta,
            ),
        )
        rows.append(rr)
        if print_rows:
            rg = "-.-" if reg == 0 else f"{math.log10(reg):4.1f}"
            print(
                f"{outer_iteration + k:3d}r "
                f"{obj:14.7e} {orig_inf:9.2e} "
                f"{0.0:9.2e} {math.log10(mu):5.1f} {rg:>5} "
                f"{0.0:9.2e} {alpha:9.2e} {1 if k else 0:2d}{typ}",
                flush=True,
            )
        return rr

    row(0)

    # Native restoration may terminate only after sufficient original
    # infeasibility reduction.  At entry, do not accept the unchanged point.
    target = req_red * theta0_l1

    for k in range(1, max_iter + 1):
        # Inertia correction using the same schedule as the outer solver.
        reg = 0.0
        D = None
        for ir in range(int(_optn(opt, "max_inertia_tries", 20))):
            D = _build_direction(
                self,
                x=state["x"], u=state["u"], s=state["s"],
                lam=state["lam"], beta=state["beta"], gamma=state["gamma"],
                zL=state["zL"], zU=state["zU"],
                pd=state["pd"], nd=state["nd"],
                pe=state["pe"], ne=state["ne"],
                ph=state["ph"], nh=state["nh"],
                zpd=state["zpd"], znd=state["znd"],
                zpe=state["zpe"], zne=state["zne"],
                zph=state["zph"], znh=state["znh"],
                xref=xref, uref=uref, sref=sref,
                rho=rho, zeta=zeta, mu=mu, reg=reg,
                backend=backend, stagewise=stagewise,
            )
            if bool(D["sol"].inertia_ok):
                finite = all(
                    np.all(np.isfinite(np.asarray(q)))
                    for q in (
                        D["sol"].dx, D["du"], D["ds"],
                        D["dpd"], D["dnd"], D["dpe"], D["dne"],
                        D["dph"], D["dnh"],
                    )
                )
                if finite:
                    break
            reg = (
                float(_optn(opt, "delta_w0", 1e-4))
                if reg == 0.0
                else max(
                    float(_optn(opt, "delta_wmin", 1e-20)),
                    float(_optn(opt, "kappa_wplusem", 100.0)) * reg,
                )
            )
        else:
            return RestorationResult(
                False, state["x"], state["u"], state["s"],
                state["lam"], state["beta"], state["gamma"],
                state["zL"], state["zU"], mu, k - 1,
                "restoration_inertia_failure", rows,
            )

        ap, ad = _resto_max_step(
            self, state["s"], state["zL"], state["zU"], D, state, mu
        )
        th0 = _theta_tuple(
            _resto_constraints(
                self, state["x"], state["u"], state["s"],
                state["pd"], state["nd"], state["pe"], state["ne"],
                state["ph"], state["nh"],
            ),
            "l1",
        )
        ph0 = _resto_barrier(
            self, state["x"], state["u"], state["s"],
            state["pd"], state["nd"], state["pe"], state["ne"],
            state["ph"], state["nh"],
            xref, uref, sref, rho, zeta, mu,
        )
        dphi = _resto_dphi(
            self, D, state, xref, uref, sref, rho, zeta, mu
        )

        accepted = False
        alpha = ap
        ftype = False
        for ls in range(1, int(_optn(opt, "max_ls", 40)) + 1):
            trial = dict(state)
            trial["x"] = (state["x"] + alpha * D["sol"].dx).at[0].set(self.x0)
            trial["u"] = state["u"] + alpha * D["du"]
            trial["s"] = state["s"] + alpha * D["ds"]
            trial["lam"] = state["lam"] + alpha * D["sol"].dlambda_dyn
            trial["beta"] = state["beta"] + alpha * D["sol"].dbeta_eq
            trial["gamma"] = state["gamma"] + alpha * D["sol"].dgamma_ineq
            trial["zL"] = state["zL"] + ad * D["dzL"]
            trial["zU"] = state["zU"] + ad * D["dzU"]

            for qn, dqn in (
                ("pd", "dpd"), ("nd", "dnd"),
                ("pe", "dpe"), ("ne", "dne"),
                ("ph", "dph"), ("nh", "dnh"),
            ):
                trial[qn] = state[qn] + alpha * D[dqn]
            for zn, dzn in (
                ("zpd", "dzpd"), ("znd", "dznd"),
                ("zpe", "dzpe"), ("zne", "dzne"),
                ("zph", "dzph"), ("znh", "dznh"),
            ):
                trial[zn] = state[zn] + ad * D[dzn]

            tres = _resto_constraints(
                self, trial["x"], trial["u"], trial["s"],
                trial["pd"], trial["nd"], trial["pe"], trial["ne"],
                trial["ph"], trial["nh"],
            )
            tht = _theta_tuple(tres, "l1")
            pht = _resto_barrier(
                self, trial["x"], trial["u"], trial["s"],
                trial["pd"], trial["nd"], trial["pe"], trial["ne"],
                trial["ph"], trial["nh"],
                xref, uref, sref, rho, zeta, mu,
            )
            ok, ftype = filt.trial(tht, pht, th0, ph0, alpha, dphi)
            if ok:
                state = trial
                accepted = True
                break
            alpha *= float(_optn(opt, "gamma_alpha", 0.5))

        if not accepted:
            return RestorationResult(
                False, state["x"], state["u"], state["s"],
                state["lam"], state["beta"], state["gamma"],
                state["zL"], state["zU"], mu, k - 1,
                "restoration_line_search_failure", rows,
            )

        if not ftype:
            filt.add(th0, ph0)
        row(k, alpha, "f" if ftype else "h", reg)

        # Convergence check is on the ORIGINAL problem.  FATROP/Ipopt requires
        # at least required_infeasibility_reduction (default 0.9) before
        # leaving restoration.
        rd, re, rh = self._constraints(
            state["x"], state["u"], state["s"]
        )
        orig_l1 = self._theta(rd, re, rh, "l1")
        if orig_l1 <= target:
            lam_o, beta_o, gamma_o, zL_o, zU_o = _project_original_handoff(
                self,
                x=state["x"], u=state["u"], s=state["s"],
                x0=xstart, u0=ustart, s0=sstart,
                zL0=zLstart, zU0=zUstart,
                mu_orig=mu_orig, backend=backend,
                lammax=float(_optn(opt, "lammax", 1e3)),
                stagewise=stagewise,
            )
            return RestorationResult(
                True,
                state["x"], state["u"], state["s"],
                lam_o, beta_o, gamma_o, zL_o, zU_o,
                mu, k, "success", rows,
            )

        # Monotone barrier update for the nested restoration IPM.
        # Keep this conservative; the first Fingers episode normally exits
        # before many inner barrier reductions.
        Dnext = _build_direction(
            self,
            x=state["x"], u=state["u"], s=state["s"],
            lam=state["lam"], beta=state["beta"], gamma=state["gamma"],
            zL=state["zL"], zU=state["zU"],
            pd=state["pd"], nd=state["nd"],
            pe=state["pe"], ne=state["ne"],
            ph=state["ph"], nh=state["nh"],
            zpd=state["zpd"], znd=state["znd"],
            zpe=state["zpe"], zne=state["zne"],
            zph=state["zph"], znh=state["znh"],
            xref=xref, uref=uref, sref=sref,
            rho=rho, zeta=zeta, mu=mu, reg=0.0,
            backend=backend, stagewise=stagewise,
        )
        em, _, _ = _resto_error(self, Dnext, state, rho, mu)
        while (
            em <= float(_optn(opt, "kappa_eta", 10.0)) * mu
            and mu > float(_optn(opt, "mu_min", 1e-11))
        ):
            mun = max(
                float(_optn(opt, "mu_min", 1e-11)),
                min(
                    float(_optn(opt, "kappa_mu", 0.2)) * mu,
                    mu ** float(_optn(opt, "theta_mu", 1.5)),
                ),
            )
            if mun >= mu * (1.0 - 1e-15):
                break
            mu = mun
            # Do not recenter p/n or duals; native nested IPM carries state.
            break

    return RestorationResult(
        False, state["x"], state["u"], state["s"],
        state["lam"], state["beta"], state["gamma"],
        state["zL"], state["zU"], mu, max_iter,
        "restoration_max_iter", rows,
    )


def primal_dual_error(
    self, lin, x, u, s, lam, beta, gamma, zL, zU, mu
):
    """Mean-L1 primal-dual error used by FATROP soft restoration."""
    rx, ru, rxf = self._stationarity(lin, lam, beta, gamma)
    rd, re, rh = self._constraints(x, u, s)
    terms = [
        _meanabs(rx), _meanabs(ru), _meanabs(rxf),
        _meanabs(rd), _meanabs(re), _meanabs(rh),
    ]
    if self.ni:
        flo, fhi, Sl, Su = _finite_margins(self, s)
        rs = -gamma - zL + zU
        terms.append(_meanabs(rs))
        terms.append(_meanabs(jnp.where(flo, Sl * zL - mu, 0.0)))
        terms.append(_meanabs(jnp.where(fhi, Su * zU - mu, 0.0)))
    return float(sum(terms))
