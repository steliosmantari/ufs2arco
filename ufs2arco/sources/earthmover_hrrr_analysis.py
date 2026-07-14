import logging
import os
from typing import Optional

import pandas as pd
import xarray as xr

from ufs2arco.sources import CloudZarrData, Source

logger = logging.getLogger("ufs2arco")


class EarthmoverHRRRAnalysis(CloudZarrData, Source):
    """
    NOAA HRRR analysis (fhr=0) served as an analysis-ready Icechunk store from
    Earthmover's Arraylake catalog, e.g.
    ``mantari-industries/noaa-hrrr-analysis-subscription``.

    This is a time-based (reanalysis-style) source: it has no forecast-hour
    dimension, so the anemoi target treats it exactly like the GCS ERA5 / replay
    sources (``_has_fhr`` is False, ``datetime == source.time``). The resulting
    anemoi zarr is structurally identical to the one produced from the GRIB2
    ``aws_hrrr_archive`` source at fhr=0, only without the GRIB download + cfgrib
    decode step.

    Access is via the arraylake client + an Icechunk read-only session rather than
    an anonymous zarr URL, so we override how the store is opened but inherit
    :meth:`CloudZarrData.open_sample_dataset` and the variable/level subsetting.

    Note:
        ``rename``, ``static_vars`` and ``horizontal_dims`` all default to the
        native HRRR CONUS assumptions but are overridable from the recipe, since
        the exact schema of the subscription is set on the Earthmover side. Run
        ``probe_earthmover_hrrr.py`` first and reconcile the ``rename`` block with
        what it reports.
    """

    sample_dims = ("time",)
    horizontal_dims = ("y", "x")
    static_vars = ("lsm", "orog")

    # Best-guess mapping from the subscription's names to ufs2arco canonical
    # names. Override from the recipe (`source.rename:`) once the probe confirms
    # the actual variable/coordinate names in the store.
    _default_rename = {
        "isobaricInhPa": "level",
        "pressure": "level",
    }

    @property
    def rename(self) -> dict:
        return self._rename_map

    def __init__(
        self,
        time: dict,
        repo: Optional[str] = None,
        uri: Optional[str] = None,
        branch: str = "main",
        token_env: str = "ARRAYLAKE_TOKEN",
        variables: Optional[list | tuple] = None,
        levels: Optional[list | tuple] = None,
        use_nearest_levels: Optional[bool] = False,
        slices: Optional[dict] = None,
        rename: Optional[dict] = None,
        static_vars: Optional[list | tuple] = None,
        horizontal_dims: Optional[list | tuple] = None,
    ) -> None:
        """
        Args:
            time (dict): passed to ``pandas.date_range`` (start, end, freq). freq
                must align with the store's analysis cadence (HRRR analysis is 1h)
                so that exact-label ``.sel(time=...)`` succeeds per sample.
            repo (str, optional): Arraylake repo as "org/name", e.g.
                "mantari-industries/noaa-hrrr-analysis-subscription". Opened via
                the arraylake client + Icechunk read-only session.
            uri (str, optional): alternative direct zarr/icechunk URI opened with
                ``xr.open_zarr``. Exactly one of ``repo`` or ``uri`` is required.
            branch (str): Icechunk branch to read (default "main").
            token_env (str): env var holding the Earthmover API token (default
                "ARRAYLAKE_TOKEN"); if unset, the arraylake client falls back to
                its own on-disk config.
            variables, levels, use_nearest_levels, slices: see :class:`Source`.
            rename (dict, optional): store-name -> canonical-name overrides,
                merged over the built-in defaults.
            static_vars, horizontal_dims (optional): override the class defaults
                if the store's schema differs from native HRRR.
        """
        if (repo is None) == (uri is None):
            raise ValueError(
                f"{self.name}: provide exactly one of 'repo' (Arraylake org/name) "
                f"or 'uri' (direct zarr URI)."
            )

        self.time = pd.date_range(**time)
        self._rename_map = {**self._default_rename, **(rename or {})}
        if static_vars is not None:
            self.static_vars = tuple(static_vars)
        if horizontal_dims is not None:
            self.horizontal_dims = tuple(horizontal_dims)

        # 1. open the store (Arraylake/Icechunk or a direct URI)
        if repo is not None:
            xds = self._open_arraylake(repo, branch, token_env)
        else:
            xds = xr.open_zarr(uri, decode_timedelta=True)

        # 2. rename to ufs2arco canonical names, then run the same subsetting
        #    CloudZarrData does (we can't call its __init__ because it hardcodes
        #    an anonymous open).
        self._xds = xds.rename({k: v for k, v in self.rename.items() if k in xds})

        Source.__init__(
            self,
            variables=variables,
            levels=levels,
            use_nearest_levels=use_nearest_levels,
            slices=slices,
        )

        self._xds = self._xds[self.variables]
        if self.levels is not None:
            self._xds = self._xds.sel(level=self.levels, **self._level_sel_kwargs)
        self._xds = self.apply_slices(self._xds)

    @staticmethod
    def _open_arraylake(repo: str, branch: str, token_env: str) -> xr.Dataset:
        try:
            from arraylake import Client
        except ImportError as e:
            raise ImportError(
                "EarthmoverHRRRAnalysis requires the 'arraylake' (and 'icechunk') "
                "packages. Install with: pip install arraylake icechunk"
            ) from e

        token = os.environ.get(token_env)
        client = Client(token=token) if token else Client()
        logger.info(f"EarthmoverHRRRAnalysis: opening Arraylake repo {repo} (branch={branch})")
        al_repo = client.get_repo(repo)
        session = al_repo.readonly_session(branch=branch)
        # zarr v3 Icechunk store; icechunk has no consolidated metadata.
        return xr.open_zarr(session.store, consolidated=False, decode_timedelta=True)
