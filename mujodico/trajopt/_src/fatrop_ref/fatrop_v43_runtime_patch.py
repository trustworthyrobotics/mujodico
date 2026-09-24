#!/usr/bin/env python3
"""Runtime v43 patch: keep FATROP restoration dynamics hard.

Prerequisite: the installed restoration_v32.py must be the validated v37
paired-elastic version (v41 reinstalls it).  This module does not overwrite
solver.py or restoration_v32.py.  It monkey-patches only the current process.

V42 established experimentally that:
  * v37 du/contact-force displacement already matches native very closely;
  * v37 dx is wrong almost entirely because of one state component;
  * enforcing A dx + B du + rd = 0 with the same du gives native-like dx.

v43 therefore removes ONLY the dynamics p_d/n_d elastic pair from the actual
structured Newton solve.  Local contact equalities and path rows retain the
validated v37 paired-elastic condensation.
"""
from __future__ import annotations

import inspect
import types

import jax
import jax.numpy as jnp


def install_v43_hard_dynamics(rv):
    src = inspect.getsource(rv._build_direction)
    if "pair_reduce" not in src or "pair_heff_dyn" not in src:
        raise RuntimeError(
            "v43 expects the installed v37 paired-elastic restoration. "
            "The current restoration_v32.py does not look like v37. "
            "Run the v37 installer from the earlier bundle first."
        )

    original_constraints = rv._resto_constraints
    original_build = rv._build_direction

    def hard_resto_constraints(
        self, x, u, s, pd, nd, pe, ne, ph, nh
    ):
        # Original NLP residuals already include h(x,u)-s in the third block.
        rd, re, rh = self._constraints(x, u, s)
        # Dynamics are HARD in v43: no pd/nd relaxation.
        return rd, re + pe - ne, rh + ph - nh

    def hard_build_direction(
        self, *, x, u, s, lam, beta, gamma, zL, zU,
        pd, nd, pe, ne, ph, nh,
        zpd, znd, zpe, zne, zph, znh,
        xref, uref, sref, rho, zeta, mu, reg, backend, stagewise,
    ):
        """Exact v37 restoration KKT with the dynamics elastic pair removed."""
        lin0 = self._linearize(x, u, stagewise)
        lin_active = self._active_hessian_lin(
            lin0, x, u, lam, beta, gamma
        )

        (
            gx0, gu0, Q0, S0, R0, rd, A, B,
            re, E, F, h, Gx, Gu, gf0, Qf0
        ) = lin0
        (
            _, _, Qa, Sa, Ra, _, _, _, _, _, _, _, _, _, _, Qfa
        ) = lin_active

        dtype = x.dtype
        N, nx, nu, neq, ni = (
            self.N, self.nx, self.nu, self.ne, self.ni
        )

        drx = 1.0 / jnp.maximum(1.0, jnp.abs(xref))
        dru = 1.0 / jnp.maximum(1.0, jnp.abs(uref))
        drs = 1.0 / jnp.maximum(1.0, jnp.abs(sref))

        # Objective scale is zero in restoration; retain only nonlinear
        # constraint curvature plus the restoration proximity Hessian.
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

        # Original bounded inequality slack elimination, unchanged from v37.
        if ni:
            flo, fhi, Sl, Su = rv._finite_margins(self, s)
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

        # Keep v37 paired elastic variables for local equalities and paths.
        qpe = rho + beta - mu / pe if neq else pe
        qne = rho - beta - mu / ne if neq else ne
        qph = rho + gamma - mu / ph if ni else ph
        qnh = rho - gamma - mu / nh if ni else nh

        hp_e = zpe / pe if neq else pe
        hn_e = zne / ne if neq else ne
        hp_h = zph / ph if ni else ph
        hn_h = znh / nh if ni else nh

        def pair_reduce(hp, hn, rp, rn):
            wp = 1.0 / hp
            wn = 1.0 / hn
            w = wp + wn
            heff = 1.0 / w
            sqrt_h = jnp.sqrt(heff)
            col = 1.0 / sqrt_h
            rscaled = (
                heff * (rp * wp - rn * wn)
            ) / sqrt_h
            return col, rscaled, heff

        if neq:
            col_e, rt_e, heff_e = pair_reduce(
                hp_e, hn_e, qpe, qne
            )
        else:
            col_e = jnp.zeros((N, 0), dtype=dtype)
            rt_e = col_e
            heff_e = col_e

        if ni:
            col_h, rt_h, heff_h = pair_reduce(
                hp_h, hn_h, qph, qnh
            )
        else:
            col_h = jnp.zeros((N, 0), dtype=dtype)
            rt_h = col_h
            heff_h = col_h

        # Augmented stage variable: [du | t_eq | t_path].
        # There is deliberately NO t_dyn block.
        off_e = nu
        off_h = off_e + neq
        nvr = off_h + ni

        R = jnp.zeros((N, nvr, nvr), dtype=dtype)
        R = R.at[:, :nu, :nu].set(Rorig)

        if neq:
            R = R.at[:, off_e:off_h, off_e:off_h].set(
                jnp.broadcast_to(
                    jnp.eye(neq, dtype=dtype), (N, neq, neq)
                )
            )
        if ni:
            R = R.at[:, off_h:, off_h:].set(
                jnp.broadcast_to(
                    jnp.eye(ni, dtype=dtype), (N, ni, ni)
                )
            )

        Saug = jnp.zeros((N, nx, nvr), dtype=dtype)
        Saug = Saug.at[:, :, :nu].set(S)

        parts = [ru]
        if neq:
            parts.append(rt_e)
        if ni:
            parts.append(rt_h)
        raug = jnp.concatenate(parts, axis=1)

        # HARD dynamics: only the physical B matrix appears.
        Baug = jnp.zeros((N, nx, nvr), dtype=dtype)
        Baug = Baug.at[:, :, :nu].set(B)

        Faug = jnp.zeros((N, neq, nvr), dtype=dtype)
        if neq:
            Faug = Faug.at[:, :, :nu].set(F)
            Ie = jnp.broadcast_to(
                jnp.eye(neq, dtype=dtype), (N, neq, neq)
            )
            Faug = Faug.at[:, :, off_e:off_h].set(
                Ie * col_e[:, None, :]
            )

        Guaug = jnp.zeros((N, ni, nvr), dtype=dtype)
        if ni:
            Guaug = Guaug.at[:, :, :nu].set(Gu)
            Ih = jnp.broadcast_to(
                jnp.eye(ni, dtype=dtype), (N, ni, ni)
            )
            Guaug = Guaug.at[:, :, off_h:].set(
                Ih * col_h[:, None, :]
            )

        # Dynamics residual is NOT relaxed.
        ard = rd
        ere = re + pe - ne
        rrh = h - s + ph - nh if ni else h
        gt = rrh + wd * ss if ni else rrh

        fn = rv.solve_serial if backend == "serial" else rv.solve_pas
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
        dbeta = sol.dbeta_eq
        dgamma = sol.dgamma_ineq

        # No dynamics elastic variables in v43.
        dpd = jnp.zeros_like(pd)
        dnd = jnp.zeros_like(nd)

        # Recover local/path p/n directions exactly as in v37.
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

        if ni:
            ds = (
                jnp.einsum("nij,nj->ni", Gx, sol.dx[:-1])
                + jnp.einsum("nij,nj->ni", Gu, du)
                + dph - dnh + rrh
            )
            dzL = jnp.where(
                flo, (-rL - zL * ds) / Sl, 0.0
            )
            dzU = jnp.where(
                fhi, (-rU + zU * ds) / Su, 0.0
            )
        else:
            ds = s
            dzL = s
            dzU = s

        def dz_lower(q, zq, dq):
            comp = q * zq - mu
            return (-comp - zq * dq) / q

        # Frozen placeholders for the removed dynamics elastic pair.
        dzpd = jnp.zeros_like(zpd)
        dznd = jnp.zeros_like(znd)

        dzpe = dz_lower(pe, zpe, dpe) if neq else pe
        dzne = dz_lower(ne, zne, dne) if neq else ne
        dzph = dz_lower(ph, zph, dph) if ni else ph
        dznh = dz_lower(nh, znh, dnh) if ni else nh

        return dict(
            sol=sol, du=du, ds=ds, dzL=dzL, dzU=dzU,
            dpd=dpd, dnd=dnd,
            dpe=dpe, dne=dne, dph=dph, dnh=dnh,
            dzpd=dzpd, dznd=dznd,
            dzpe=dzpe, dzne=dzne,
            dzph=dzph, dznh=dznh,
            lin=lin0,
            pair_heff_dyn=jnp.zeros((N, 0), dtype=dtype),
            pair_heff_eq=heff_e,
            pair_heff_path=heff_h,
        )

    rv._resto_constraints = hard_resto_constraints
    rv._build_direction = hard_build_direction

    return {
        "original_constraints": original_constraints,
        "original_build": original_build,
        "mode": "v43_hard_dynamics",
    }
