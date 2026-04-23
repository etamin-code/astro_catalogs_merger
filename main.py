from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Iterable
import io
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u


DATA_DIR = Path("data")
C_KMS = 299792.458
H0_DEFAULT = 70.0


# =========================================================
# Helpers
# =========================================================

def to_numeric_safe(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def velocity_to_redshift(velocity_kms: pd.Series | float) -> pd.Series | float:
    return velocity_kms / C_KMS


def distance_to_redshift_hubble(
    distance_mpc: pd.Series | float,
    h0: float = H0_DEFAULT,
) -> pd.Series | float:
    return (h0 * distance_mpc) / C_KMS


def hms_dms_to_deg(
    ra_series: pd.Series,
    dec_series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    ra_out = pd.Series(np.nan, index=ra_series.index, dtype=float)
    dec_out = pd.Series(np.nan, index=dec_series.index, dtype=float)

    mask = ra_series.notna() & dec_series.notna()
    if mask.any():
        coords = SkyCoord(
            ra_series[mask].astype(str).str.strip().values,
            dec_series[mask].astype(str).str.strip().values,
            unit=(u.hourangle, u.deg),
            frame="icrs",
        )
        ra_out.loc[mask] = coords.ra.deg
        dec_out.loc[mask] = coords.dec.deg

    return ra_out, dec_out


def extract_t_from_morph_string(series: pd.Series | None) -> pd.Series:
    """
    Converts morphology string classifications into numerical T-type classes.

    Supports morphology labels such as:
    E, S0, S0/a, SA, SAB, SB, SBC, SC, SC-IRR, IRR.

    Returns:
        pd.Series of numerical T values.
    """
    if series is None:
        return pd.Series(dtype=float)

    morphology = (
        series.astype(str)
        .str.upper()
        .str.strip()
    )

    def map_morph_to_t(value: str) -> float:
        if value.startswith("E"):
            return -5

        elif value.startswith("S0/A"):
            return 0

        elif value.startswith("S0"):
            return -2

        elif value.startswith("SAB"):
            return 2

        elif value.startswith("SBC"):
            return 4

        elif value.startswith("SB"):
            return 3

        elif value.startswith("SA"):
            return 1

        elif value.startswith("SC-IRR"):
            return 8

        elif value.startswith("SC"):
            return 5

        elif value.startswith("IRR"):
            return 9

        return float("nan")

    return morphology.apply(map_morph_to_t)

def make_z_origin_series(
    index,
    direct_mask: pd.Series | None = None,
    velocity_mask: pd.Series | None = None,
    distance_mask: pd.Series | None = None,
) -> pd.Series:
    origin = pd.Series(None, index=index, dtype="object")

    if distance_mask is not None:
        origin.loc[distance_mask.fillna(False)] = "distance"

    if velocity_mask is not None:
        origin.loc[velocity_mask.fillna(False)] = "velocity"

    if direct_mask is not None:
        origin.loc[direct_mask.fillna(False)] = "direct"

    return origin


def angular_sep_arcsec(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    ra1 = np.deg2rad(ra1_deg)
    dec1 = np.deg2rad(dec1_deg)
    ra2 = np.deg2rad(ra2_deg)
    dec2 = np.deg2rad(dec2_deg)

    dra = ra2 - ra1
    dra = (dra + np.pi) % (2 * np.pi) - np.pi
    ddec = dec2 - dec1

    a = np.sin(ddec / 2.0) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return np.rad2deg(c) * 3600.0


# =========================================================
# Base configurable reader
# =========================================================

class CatalogReader(ABC):
    STANDARD_COLUMNS = [
        "name",
        "ra_deg",
        "dec_deg",
        "t_type",
        "redshift",
        "distance_mpc",
        "z_origin",
        "source",
        "catalogs",
    ]

    # ---- config to override in child classes ----
    NAME_COL: str | None = None

    RA_COL: str | None = None
    DEC_COL: str | None = None
    COORD_FORMAT: str = "deg"   # "deg" | "hms_dms"

    T_COL: str | None = None
    T_SOURCE: str | None = None  # "numeric" | "extract_from_morph" | None

    Z_COL: str | None = None
    VELOCITY_COL: str | None = None
    DISTANCE_COL: str | None = None


    def __init__(self, path: str | Path, source_name: str):
        self.path = Path(path)
        self.source_name = source_name
        self.raw_df: pd.DataFrame | None = None

    # ---------------------------
    # Public API
    # ---------------------------

    def read(self) -> pd.DataFrame:
        self.raw_df = self._read_vizier_tsv(self.path)
        prepared = self.to_unified_df()
        prepared = self._finalize(prepared)
        return prepared

    def size(self) -> int:
        if self.raw_df is None:
            self.raw_df = self._read_vizier_tsv(self.path)
        return len(self.raw_df)

    def get_raw_df(self) -> pd.DataFrame:
        if self.raw_df is None:
            self.raw_df = self._read_vizier_tsv(self.path)
        return self.raw_df

    # ---------------------------
    # Universal unified builder
    # ---------------------------

    def to_unified_df(self) -> pd.DataFrame:
        df = self.get_raw_df()

        out = pd.DataFrame(index=df.index)
        out["name"] = self.get_name()
        out["ra_deg"] = self.get_ra_deg()
        out["dec_deg"] = self.get_dec_deg()
        out["t_type"] = self.get_t_type()
        out["distance_mpc"] = self.get_distance_mpc()
        out["redshift"] = self.get_redshift()
        out["z_origin"] = self.get_z_origin()

        if out["t_type"].isna().all():
            print(
                f"[INFO] {self.__class__.__name__}: morphology is absent or could not be parsed, "
                f"morphology filtering skipped. Num of rows = {len(out)}"
            )
            return out

        filtered_by_morph = out[out["t_type"].isin([2, 3])].copy()

        print(
            f"[INFO] {self.__class__.__name__}: morphology filter applied "
            f"(types 2 and 3), kept {len(filtered_by_morph)} of {len(out)} rows."
        )

        return filtered_by_morph

    # ---------------------------
    # Universal getters
    # ---------------------------

    def get_name(self) -> pd.Series:
        return self._get_column(self.NAME_COL)

    def get_ra_deg(self) -> pd.Series:
        if self.RA_COL is None or self.DEC_COL is None:
            return pd.Series(np.nan, index=self.get_raw_df().index, dtype=float)

        if self.COORD_FORMAT == "deg":
            return self._get_numeric_column(self.RA_COL)

        if self.COORD_FORMAT == "hms_dms":
            ra, _ = hms_dms_to_deg(
                self._get_column(self.RA_COL),
                self._get_column(self.DEC_COL),
            )
            return ra

        raise ValueError(f"Unsupported COORD_FORMAT={self.COORD_FORMAT}")

    def get_dec_deg(self) -> pd.Series:
        if self.RA_COL is None or self.DEC_COL is None:
            return pd.Series(np.nan, index=self.get_raw_df().index, dtype=float)

        if self.COORD_FORMAT == "deg":
            return self._get_numeric_column(self.DEC_COL)

        if self.COORD_FORMAT == "hms_dms":
            _, dec = hms_dms_to_deg(
                self._get_column(self.RA_COL),
                self._get_column(self.DEC_COL),
            )
            return dec

        raise ValueError(f"Unsupported COORD_FORMAT={self.COORD_FORMAT}")

    def get_t_type(self) -> pd.Series:
        if self.T_COL is None or self.T_SOURCE is None:
            return pd.Series(np.nan, index=self.get_raw_df().index, dtype=float)

        if self.T_SOURCE == "numeric":
            return self._get_numeric_column(self.T_COL)

        if self.T_SOURCE == "extract_from_morph":
            return extract_t_from_morph_string(self._get_column(self.T_COL))

        raise ValueError(f"Unsupported T_SOURCE={self.T_SOURCE}")

    def get_distance_mpc(self) -> pd.Series:
        if self.DISTANCE_COL is None:
            return pd.Series(np.nan, index=self.get_raw_df().index, dtype=float)
        return self._get_numeric_column(self.DISTANCE_COL)

    def get_redshift(self) -> pd.Series:
        if self.Z_COL is not None:
            z = self._get_numeric_column(self.Z_COL)
            if z.notna().any():
                return z

        if self.VELOCITY_COL is not None:
            vel = self._get_numeric_column(self.VELOCITY_COL)
            z_from_vel = velocity_to_redshift(vel)

            if self.DISTANCE_COL is not None:
                dist = self.get_distance_mpc()
                return z_from_vel.fillna(distance_to_redshift_hubble(dist))

            return z_from_vel

        if self.DISTANCE_COL is not None:
            dist = self.get_distance_mpc()
            return distance_to_redshift_hubble(dist)

        return pd.Series(np.nan, index=self.get_raw_df().index, dtype=float)

    def get_z_origin(self) -> pd.Series:
        z = self._get_numeric_column(self.Z_COL) if self.Z_COL is not None else None
        vel = self._get_numeric_column(self.VELOCITY_COL) if self.VELOCITY_COL is not None else None
        dist = self._get_numeric_column(self.DISTANCE_COL) if self.DISTANCE_COL is not None else None

        return make_z_origin_series(
            index=self.get_raw_df().index,
            direct_mask=z.notna() if z is not None else None,
            velocity_mask=vel.notna() if vel is not None else None,
            distance_mask=dist.notna() if dist is not None else None,
        )

    # ---------------------------
    # Column helpers
    # ---------------------------

    def _get_column(self, name: str | None) -> pd.Series:
        df = self.get_raw_df()
        if name is None or name not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype="object")
        return df[name]

    def _get_numeric_column(self, name: str | None) -> pd.Series:
        return to_numeric_safe(self._get_column(name))

    # ---------------------------
    # Generic TSV reader
    # ---------------------------

    @staticmethod
    def _is_separator_row(parts: list[str]) -> bool:
        non_empty = [p.strip() for p in parts if p.strip()]
        if not non_empty:
            return False
        return all(set(token) <= {"-"} for token in non_empty)

    @staticmethod
    def _is_units_row(parts: list[str]) -> bool:
        known_tokens = {
            "deg", "arcsec", "arcmin", "mag", "km/s", "Mpc", "kpc", "Jy",
            "mJy", "Lsun", "s", "ks", "ct/ks", "Msun/yr",
            '"h:m:s"', '"d:m:s"', "h:m:s", "d:m:s",
            "[Msun]", "[Msun/yr]", "[W]", "[10-7W]", "[yr-1]",
            "", "---"
        }
        non_empty = [p.strip() for p in parts if p.strip()]
        if not non_empty:
            return False

        short_like = 0
        for token in non_empty:
            if token in known_tokens or len(token) <= 10:
                short_like += 1

        return short_like / max(len(non_empty), 1) > 0.7

    def _read_vizier_tsv(self, path: Path) -> pd.DataFrame:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

        non_comment_lines = [line for line in lines if not line.startswith("#") and line.strip()]
        if not non_comment_lines:
            raise ValueError(f"No data content found in {path}")

        header_line = non_comment_lines[0]
        data_lines = []

        for i, line in enumerate(non_comment_lines[1:], start=1):
            parts = line.split("\t")

            if i == 1 and self._is_units_row(parts):
                continue

            if self._is_separator_row(parts):
                continue

            data_lines.append(line)

        buffer = io.StringIO(header_line + "\n" + "\n".join(data_lines))
        df = pd.read_csv(buffer, sep="\t", dtype=str)

        df = df.dropna(axis=1, how="all")

        if len(df.columns) > 0 and str(df.columns[0]).startswith("Unnamed"):
            df = df.iloc[:, 1:]

        df.columns = [str(c).strip() for c in df.columns]
        return df

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.STANDARD_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        df["source"] = self.source_name
        df["catalogs"] = [[self.source_name] for _ in range(len(df))]

        for col in ["ra_deg", "dec_deg", "t_type", "redshift", "distance_mpc"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[self.STANDARD_COLUMNS].copy()


class TwoMigReader(CatalogReader):
    NAME_COL = "Name"
    RA_COL = "_RAJ2000"
    DEC_COL = "_DEJ2000"
    COORD_FORMAT = "deg"
    T_COL = "T"
    T_SOURCE = "numeric"
    VELOCITY_COL = "HRV"


class Mwa2MigBarredReader(CatalogReader):
    NAME_COL = "Name"
    RA_COL = "RAJ2000, deg"
    DEC_COL = "DEJ2000, deg"
    COORD_FORMAT = "deg"
    T_COL = "T, Morphological type in de Vaucouleurs' scale"
    T_SOURCE = "numeric"
    Z_COL = "Redshift"
    VELOCITY_COL = "HRV, km/s"


class Bai2015Reader(CatalogReader):
    NAME_COL = "PGC"
    RA_COL = "RAJ2000"
    DEC_COL = "DEJ2000"
    COORD_FORMAT = "hms_dms"
    T_COL = "MT"
    T_SOURCE = "numeric"
    DISTANCE_COL = "Dist"


class Bi2020Reader(CatalogReader):
    NAME_COL = "Name"
    RA_COL = "RAJ2000"
    DEC_COL = "DEJ2000"
    T_COL = "HType"
    COORD_FORMAT = "hms_dms"
    T_SOURCE = "extract_from_morph"


    # тут нестандартно, тому лише це перевизначаємо
    def get_distance_mpc(self) -> pd.Series:
        avg_dist = self._get_numeric_column("AvgDist")
        g_dist = self._get_numeric_column("GDist")
        dist = self._get_numeric_column("Dist")
        return avg_dist.fillna(g_dist).fillna(dist)


class Boardman2020Reader(CatalogReader):
    NAME_COL = "MaNGA ID"
    RA_COL = "RA (deg)"
    DEC_COL = "DEC (deg)"
    COORD_FORMAT = "deg"
    Z_COL = "z"


class Fraser2019Reader(CatalogReader):
    NAME_COL = "Name"
    RA_COL = "RAJ2000, deg"
    DEC_COL = "DEJ2000, deg"
    COORD_FORMAT = "deg"


class Galliano2021Reader(CatalogReader):
    NAME_COL = "Galaxy"
    RA_COL = "_RA"
    DEC_COL = "_DE"
    COORD_FORMAT = "deg"


class Heesen2023Reader(CatalogReader):
    NAME_COL = "Galaxy"
    RA_COL = "_RA"
    DEC_COL = "_DE"
    COORD_FORMAT = "deg"
    DISTANCE_COL = "Dist"


class Ofek2017Reader(CatalogReader):
    NAME_COL = "Target"
    RA_COL = "RAJ2000"
    DEC_COL = "DEJ2000"
    COORD_FORMAT = "hms_dms"
    Z_COL = "z"


class Ohlson2024Reader(CatalogReader):
    NAME_COL = "Name"
    RA_COL = "RAJ2000"
    DEC_COL = "DEJ2000"
    COORD_FORMAT = "deg"
    T_COL = "TType"
    T_SOURCE = "numeric"
    DISTANCE_COL = "bestDist"
    VELOCITY_COL = "HRV"


class Paspaliaris2025Reader(CatalogReader):
    NAME_COL = "Id"
    RA_COL = "RAJ2000"
    DEC_COL = "DEJ2000"
    COORD_FORMAT = "hms_dms"
    T_COL = "Type"
    DISTANCE_COL = "D"
    T_SOURCE = "extract_from_morph"



class PilyuginReader(CatalogReader):
    NAME_COL = "Name"
    RA_COL = "RA (deg)"
    DEC_COL = "DEC (deg)"
    COORD_FORMAT = "deg"
    T_COL = "Morphological Type"
    T_SOURCE = "extract_from_morph"
    DISTANCE_COL = "Distance (Mpc)"


class Schmidt1993Reader(CatalogReader):
    NAME_COL = "Name"
    RA_COL = "_RA.icrs"
    DEC_COL = "_DE.icrs"
    COORD_FORMAT = "hms_dms"
    T_COL = "type"
    T_SOURCE = "numeric"
    VELOCITY_COL = "HVel"


class Tully1992Reader(CatalogReader):
    NAME_COL = "Gal"
    DISTANCE_COL = "Dist"
    VELOCITY_COL = "Vel"

    # координат немає, тому окремо не треба нічого робити


class Wakker2015_1Reader(CatalogReader):
    NAME_COL = "Target"
    RA_COL = "RAJ2000"
    DEC_COL = "DEJ2000"
    COORD_FORMAT = "hms_dms"
    Z_COL = "z"


class Wakker2015_2Reader(CatalogReader):
    NAME_COL = "Gal"
    RA_COL = "_RA"
    DEC_COL = "_DE"
    COORD_FORMAT = "deg"
    VELOCITY_COL = "cz"
    T_COL = "Type"
    T_SOURCE = "extract_from_morph"

CATALOG_READER_REGISTRY = {
    # "2mig.tsv": (TwoMigReader, "2mig"),
    "MWA-2MIG-barred.tsv": (Mwa2MigBarredReader, "mwa_2mig_barred"),
    "Bai2015.tsv": (Bai2015Reader, "bai2015"),
    "Bi2020.tsv": (Bi2020Reader, "bi2020"),
    "Boardman2020 candidates.tsv": (Boardman2020Reader, "boardman2020"),
    "fraser2019 candidates.tsv": (Fraser2019Reader, "fraser2019"),
    "Galliano2021.tsv": (Galliano2021Reader, "galliano2021"),
    "Heesen2023.tsv": (Heesen2023Reader, "heesen2023"),
    "Ofek2017.tsv": (Ofek2017Reader, "ofek2017"),
    "Ohlson2024.tsv": (Ohlson2024Reader, "ohlson2024"),
    "Paspaliaris2025.tsv": (Paspaliaris2025Reader, "paspaliaris2025"),
    "Pilyugin.tsv": (PilyuginReader, "pilyugin"),
    "Schmidt1993.tsv": (Schmidt1993Reader, "schmidt1993"),
    "Tully1992.tsv": (Tully1992Reader, "tully1992"),
    "Wakker2015_1.tsv": (Wakker2015_1Reader, "wakker2015_1"),
    "Wakker2015_2.tsv": (Wakker2015_2Reader, "wakker2015_2"),
}


def get_catalog_reader(path: str | Path) -> CatalogReader:
    path = Path(path)
    fname = path.name

    if fname not in CATALOG_READER_REGISTRY:
        raise KeyError(f"No reader registered for file: {fname}")

    reader_cls, source_name = CATALOG_READER_REGISTRY[fname]
    return reader_cls(path=path, source_name=source_name)


class CatalogMerger:
    def __init__(self, dedup_threshold_arcsec: float = 2.0):
        self.dedup_threshold_arcsec = dedup_threshold_arcsec

    def merge_catalogs(self, catalog_paths: Iterable[str | Path]) -> pd.DataFrame:
        loaded_catalogs: list[tuple[str, pd.DataFrame]] = []

        # read all
        for path in catalog_paths:
            reader = get_catalog_reader(path)
            df = reader.read()

            # useful cleanup
            df = df[df["name"].notna() | (df["ra_deg"].notna() & df["dec_deg"].notna())].copy()
            loaded_catalogs.append((reader.source_name, df))

        # sort by size descending
        loaded_catalogs.sort(key=lambda x: len(x[1]), reverse=True)

        united = pd.DataFrame(columns=CatalogReader.STANDARD_COLUMNS)

        for source_name, df_new in loaded_catalogs:
            if united.empty:
                united = df_new.copy().reset_index(drop=True)
                continue

            united = self._merge_one_catalog(united, df_new, source_name)

        # make catalogs readable
        united["catalogs"] = united["catalogs"].apply(
            lambda x: ",".join(sorted(set(x))) if isinstance(x, list) else x
        )

        return united

    def _merge_one_catalog(
        self,
        united: pd.DataFrame,
        df_new: pd.DataFrame,
        source_name: str,
    ) -> pd.DataFrame:
        keep_rows = []
        num_of_duplicates = 0
        for _, row in df_new.iterrows():
            is_dup, dup_idx = self._find_duplicate_index(united, row)

            if is_dup:
                catalogs = united.at[dup_idx, "catalogs"]
                if not isinstance(catalogs, list):
                    catalogs = [catalogs] if pd.notna(catalogs) else []
                catalogs.append(source_name)
                united.at[dup_idx, "catalogs"] = catalogs
                num_of_duplicates += 1
                # print(f"found duplicate: {row}\n"
                #       f"with the row {united.loc[dup_idx]}")
            else:
                keep_rows.append(row)

        if keep_rows:
            united = pd.concat([united, pd.DataFrame(keep_rows)], ignore_index=True)
        print(f"found {num_of_duplicates} duplicates out of {len(df_new)} input items in {source_name}")
        return united

    def _find_duplicate_index(self, united: pd.DataFrame, row: pd.Series) -> tuple[bool, int | None]:
        if pd.isna(row["ra_deg"]) or pd.isna(row["dec_deg"]):
            return False, None

        coords_mask = united["ra_deg"].notna() & united["dec_deg"].notna()
        if not coords_mask.any():
            return False, None

        candidates = united[coords_mask]

        seps = angular_sep_arcsec(
            row["ra_deg"],
            row["dec_deg"],
            candidates["ra_deg"].to_numpy(),
            candidates["dec_deg"].to_numpy(),
        )

        min_idx_local = np.argmin(seps)
        min_sep = seps[min_idx_local]

        if min_sep < self.dedup_threshold_arcsec:
            dup_idx = candidates.index[min_idx_local]
            return True, int(dup_idx)

        return False, None


# =========================================================
# Pipeline
# =========================================================

def build_united_candidates(
    input_dir: str | Path = DATA_DIR,
    output_name: str = "United Candidates.tsv",
    dedup_threshold_arcsec: float = 10.0,
) -> pd.DataFrame:
    input_dir = Path(input_dir)

    catalog_paths = [input_dir / fname for fname in CATALOG_READER_REGISTRY.keys() if (input_dir / fname).exists()]

    merger = CatalogMerger(dedup_threshold_arcsec=dedup_threshold_arcsec)
    united = merger.merge_catalogs(catalog_paths)

    output_path = input_dir / output_name
    united.to_csv(output_path, sep="\t", index=False)

    print(f"[OK] United catalog saved to: {output_path}")
    print(f"[INFO] Final rows: {len(united)}")
    return united


if __name__ == "__main__":
    build_united_candidates()