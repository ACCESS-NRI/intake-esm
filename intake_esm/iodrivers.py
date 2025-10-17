from __future__ import annotations

import typing
from abc import abstractmethod
from typing import Self

import pandas as pd
import polars as pl
import pydantic
from pydantic import ConfigDict

__filetypes__ = ['csv', 'csv.bz2', 'csv.gz', 'csv.zip', 'csv.xz', 'parquet']


class FramesModel(pydantic.BaseModel):
    """A Pydantic model to represent our collection of dataframes - pandas, polars,
    and lazyframe."""

    df: pd.DataFrame | None = None
    pl_df: pl.DataFrame | None = None
    lf: pl.LazyFrame | None = None

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    @pydantic.model_validator(mode='after')
    def ensure_some(self) -> Self:
        """
        Make sure that at least one of the dataframes is not `None` when the model is
        instantiated.
        """
        if self.df is None and self.pl_df is None and self.lf is None:
            raise AssertionError('At least one of df, pl_df, or lf must be set')
        return self

    @property
    def pandas(self) -> pd.DataFrame:
        """Return the pandas DataFrame, instantiating it if necessary."""
        if self.df is not None:
            return self.df

        if self.pl_df is not None:
            self.df = self.pl_df.to_pandas(use_pyarrow_extension_array=False)
            self.df[list(self.columns_with_iterables)] = self.df[
                list(self.columns_with_iterables)
            ].map(tuple)
            return self.df

        self.pl_df = self.lf.collect()  # type: ignore[union-attr]
        self.df = self.pl_df.to_pandas(use_pyarrow_extension_array=False)
        for colname in self.columns_with_iterables:
            self.df[colname] = self.df[colname].apply(tuple)
        return self.df

    @property
    def polars(self) -> pl.DataFrame:
        """Return the polars DataFrame, instantiating it if necessary."""
        if self.pl_df is not None:
            return self.pl_df

        if self.lf is not None:
            self.pl_df = self.lf.collect()
            return self.pl_df

        self.pl_df = pl.from_pandas(self.df)
        self.lf = self.pl_df.lazy()

        return self.pl_df

    @property
    def lazy(self) -> pl.LazyFrame:
        """Return the polars LazyFrame, instantiating it if necessary."""
        if self.lf is not None:
            return self.lf

        # Otherwise, it must be none - so lets create the lazyframe now. We use the
        # self.polars property, so we can cascade to creating it from the pandas dataframe
        # if necessary.
        self.lf = self.polars.lazy()
        return self.lf

    @property
    def columns_with_iterables(self) -> set[str]:
        """Return a set of columns that have iterables, preferentially using
        `self.lazy` > `self.polars` > `self.pandas` to minimise overhead."""
        if (trunc_df := self.lazy.head(1).collect()).is_empty():
            return set()
        if self.df is not None and self.df.empty:
            return set()

        colnames, dtypes = trunc_df.columns, trunc_df.dtypes
        return {colname for colname, dtype in zip(colnames, dtypes) if dtype == pl.List}

    def nunique(self) -> pd.Series:
        """Return a series of the number of unique values for each column in the catalog."""
        return pd.Series(
            {
                colname: self.polars.get_column(colname).explode().n_unique()
                if self.polars.schema[colname] == pl.List
                else self.polars.get_column(colname).n_unique()
                for colname in self.polars.columns
            }
        )


class CatalogFileIoDriver:
    """Abstracts away some of the complexity related to reading dataframes"""

    def __init__(
        self,
        catalog_file: pydantic.StrictStr | None,
        storage_options: dict[str, typing.Any],
        **read_kwargs,
    ):
        self.catalog_file = catalog_file
        self.storage_options = storage_options
        self.read_kwargs = read_kwargs

        if self.catalog_file is None:
            raise AssertionError('catalog_file must be set to a valid file path or URL')

        self._dtype_map: dict[str, str] = {}
        self._frames: FramesModel | None = None

    @abstractmethod
    def read(self) -> None: ...

    @abstractmethod
    def write(self) -> None: ...

    @property
    def dtype_map(self) -> dict[str, str]:
        """Return a map of column names to their dtypes for columns with iterables."""
        if self._frames is None:
            self.read()
        return self._dtype_map

    @property
    def frames(self) -> FramesModel:
        """Return the FramesModel containing the dataframes read from the catalog file."""
        if self._frames is None:
            self.read()
        return self._frames  # type: ignore[return-value]

    @property
    @abstractmethod
    def filetype(self) -> str: ...

    @property
    @abstractmethod
    def driver(self) -> str: ...


class PolarsCsvDriver(CatalogFileIoDriver):
    """A driver to read catalog files from csv using polars"""

    def __init__(
        self,
        catalog_file: pydantic.StrictStr | None,
        storage_options: dict[str, typing.Any],
        **read_kwargs,
    ):
        super().__init__(catalog_file, storage_options, **read_kwargs)

    def read(self) -> None:
        """Read a catalog file stored as a csv using polars"""
        converters = self.read_kwargs.pop('converters', {})  # Hack
        # See https://github.com/pola-rs/polars/issues/13040 - can't use read_csv.
        lf = pl.scan_csv(
            self.catalog_file,  # type: ignore[arg-type]
            storage_options=self.storage_options,
            infer_schema=False,
            **self.read_kwargs,
        )

        if dtype_map := (
            lf.head(1)
            .select([colname for colname in converters.keys()])
            .with_columns(
                [
                    pl.col(colname)
                    .str.head(1)
                    .str.replace_many(
                        ['[', '(', '{'],
                        ['list', 'tuple', 'set'],
                    )
                    for colname in converters.keys()
                ]
            )
            .collect()
            .to_dicts()
        ):  # Returns an empty list if no rows - hence walrus
            self._dtype_map = dtype_map[0]

        lf = lf.with_columns(
            [
                pl.col(colname)
                .str.replace('^.', '[')  # Replace first/last chars with [ or ].
                .str.replace('.$', ']')  # set/tuple => list
                .str.replace(',]$', ']')  # Remove trailing commas
                .str.replace_all(
                    "'", '"'
                )  # This is to do with the JSON spec - single versus double quotes
                .str.json_decode(dtype=pl.List(pl.Utf8))
                for colname in converters.keys()
            ]
        )
        self._frames = FramesModel(lf=lf)

    def write(self) -> None:
        raise NotImplementedError('TODO')

    @property
    def filetype(self) -> str:
        return 'csv'

    @property
    def driver(self) -> str:
        return 'polars'


class PolarsParquetDriver(CatalogFileIoDriver):
    """A driver to read catalog files from parquet using polars"""

    def __init__(
        self,
        catalog_file: pydantic.StrictStr | None,
        storage_options: dict[str, typing.Any],
        **read_kwargs,
    ):
        super().__init__(catalog_file, storage_options, **read_kwargs)

    def read(self) -> None:
        """Read a catalog file stored as a parquet using polars"""
        lf = pl.scan_parquet(
            self.catalog_file,  # type: ignore[arg-type]
            storage_options=self.storage_options,
            **self.read_kwargs,
        )
        self._frames = FramesModel(lf=lf)
        self._dtype_map = {}

    def write(self) -> None:
        raise NotImplementedError('TODO')

    @property
    def filetype(self) -> str:
        return 'parquet'

    @property
    def driver(self) -> str:
        return 'polars'


class PandasCsvDriver(CatalogFileIoDriver):
    """A driver to read catalog files from csv using pandas"""

    def __init__(
        self,
        catalog_file: pydantic.StrictStr | None,
        storage_options: dict[str, typing.Any],
        **read_kwargs,
    ):
        super().__init__(catalog_file, storage_options, **read_kwargs)

    def read(self) -> None:
        """Read a catalog file stored as a csv using pandas"""
        df = pd.read_csv(
            self.catalog_file,
            storage_options=self.storage_options,
            **self.read_kwargs,
        )
        self._dtype_map = {
            colname: df['colname'].dtype
            for colname in self.read_kwargs.get('converters', {}).keys()
        }
        self._frames = FramesModel(df=df)

    def write(self) -> None:
        raise NotImplementedError('TODO')

    @property
    def filetype(self) -> str:
        return 'csv'

    @property
    def driver(self) -> str:
        return 'pandas'


class PandasParquetDriver(CatalogFileIoDriver):
    """A driver to read catalog files from csv using pandas"""

    def __init__(
        self,
        catalog_file: pydantic.StrictStr | None,
        storage_options: dict[str, typing.Any],
        **read_kwargs,
    ):
        super().__init__(catalog_file, storage_options, **read_kwargs)

    def read(self) -> None:
        raise NotImplementedError('PandasDriver does not currently support reading parquet files')

    def write(self) -> None:
        raise NotImplementedError('TODO')

    @property
    def filetype(self) -> str:
        return 'parquet'

    @property
    def driver(self) -> str:
        return 'pandas'
