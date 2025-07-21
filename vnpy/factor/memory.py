import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Self, Union

import polars as pl

from vnpy.trader.constant import Interval
from vnpy.trader.database import TimeRange
from vnpy.trader.utility import get_file_path

# To make the FactorMemory class aware of the context it's running in,
# we import FactorMode. A fallback is provided for standalone use or testing.
try:
    from vnpy.factor.base import FactorMode
except ImportError:
    from enum import Enum


    class FactorMode(Enum):
        LIVE = "live"
        BACKTEST = "backtest"


class FactorMemory:
    """
    Manages factor historical data using memory-mapped Arrow IPC files.

    This class ensures thread-safe operations and atomic writes to protect
    data integrity. It strictly enforces a schema for the stored data and
    adjusts its behavior based on the operational mode (LIVE vs. BACKTEST).

    Attributes:
        file_path (Path): Path to the Arrow IPC file.
        max_rows (int): Maximum number of rows to store.
        schema (Dict[str, pl.PolarsDataType]): Schema of the DataFrame.
        datetime_col (str): Name of the datetime column.
        mode (FactorMode): The operational mode (LIVE or BACKTEST).
        _lock (Lock): Thread lock for ensuring safe concurrent access.
    """

    def __init__(
            self,
            file_path: str | Path,
            max_rows: int,
            schema: dict[str, pl.DataType],
            datetime_col: str = "datetime",
            mode: FactorMode = FactorMode.LIVE,
    ):
        """
        Initializes the FactorMemory instance.

        Args:
            file_path: Path to the Arrow IPC file where data will be stored.
            max_rows: Maximum number of rows to maintain in the circular buffer.
            schema: A dictionary defining column names and their Polars data types.
            datetime_col: The name of the column that serves as the primary time index.
            mode (FactorMode): The operational mode, which determines cleanup behavior.
                               Defaults to LIVE.
        """
        if max_rows <= 0:
            raise ValueError("max_rows must be a positive integer.")
        if not schema:
            raise ValueError("Schema cannot be empty.")
        if datetime_col not in schema:
            raise ValueError(
                f"datetime_col '{datetime_col}' must be defined in the schema."
            )

        self.file_path = Path(file_path).resolve()
        self.max_rows = max_rows
        self.schema = schema
        self.datetime_col = datetime_col
        self.mode = mode
        self._lock = Lock()

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_if_empty()

    def _initialize_if_empty(self) -> None:
        """
        Initializes or validates the Arrow file. If the file doesn't exist,
        is empty, or has a mismatched schema, it's created/overwritten.
        """
        with self._lock:
            try:
                should_initialize = True
                if self.file_path.exists() and self.file_path.stat().st_size > 0:
                    try:
                        existing_df = pl.read_ipc(
                            self.file_path, memory_map=False, use_pyarrow=True
                        )
                        if existing_df.schema == self.schema:
                            should_initialize = False
                        else:
                            print(
                                f"Warning: File {self.file_path} exists with mismatched schema. Re-initializing."
                            )
                    except Exception as e:
                        print(
                            f"Warning: Could not read existing file {self.file_path} (Reason: {e}). Re-initializing."
                        )

                if should_initialize:
                    empty_df = pl.DataFrame(data={}, schema=self.schema)
                    self._save_data(empty_df)  # Use atomic save for initialization
                    print(
                        f"Initialized or re-initialized factor memory file: {self.file_path}"
                    )
            except Exception as e_init:
                raise OSError(
                    f"Failed to initialize factor memory file {self.file_path}: {e_init}"
                ) from e_init

    def _conform_df_to_schema(
            self, df: pl.DataFrame, df_name: str = "input"
    ) -> pl.DataFrame:
        """
        Conforms an input DataFrame to the instance's schema, ensuring
        column order, existence, and data types are correct.
        """
        if df.schema == self.schema:
            return df

        print("df.schema\n",df.schema)
        print("self.schema\n",self.schema)
        conformed_cols = {}
        errors = []
        current_df_schema = df.schema

        for col_name, expected_dtype in self.schema.items():
            if col_name in current_df_schema:
                current_dtype = current_df_schema[col_name]
                if current_dtype == expected_dtype:
                    conformed_cols[col_name] = df.get_column(col_name)
                else:
                    try:
                        conformed_cols[col_name] = df.get_column(col_name).cast(
                            expected_dtype, strict=False
                        )
                    except Exception as e:
                        errors.append(
                            f"Could not cast column '{col_name}' from {current_dtype} to {expected_dtype}: {e}")
                        conformed_cols[col_name] = pl.Series([None] * len(df), dtype=expected_dtype, name=col_name)
            else:
                errors.append(f"Column '{col_name}' missing in {df_name} DataFrame. Adding as nulls.")
                conformed_cols[col_name] = pl.Series([None] * len(df), dtype=expected_dtype, name=col_name)

        if errors:
            print(f"Warning: Schema conformance issues for {self.file_path} with {df_name} DataFrame:\n" + "\n".join(
                errors))

        return pl.DataFrame(conformed_cols).select(list(self.schema.keys()))

    def _load_data(self) -> pl.DataFrame:
        """Loads data from the Arrow file. Returns an empty DataFrame on failure."""
        if not self.file_path.exists() or self.file_path.stat().st_size == 0:
            return pl.DataFrame(data={}, schema=self.schema)
        try:
            df = pl.read_ipc(self.file_path, memory_map=True, use_pyarrow=True)
            return self._conform_df_to_schema(df, "loaded")
        except Exception as e:
            print(f"Error loading data from {self.file_path}: {e}. Returning empty DataFrame.")
            return pl.DataFrame(data={}, schema=self.schema)

    def _save_data(self, df: pl.DataFrame) -> None:
        """Saves a DataFrame to the file atomically."""
        df_to_save = self._conform_df_to_schema(df, "data_to_save")
        temp_file_path = self.file_path.with_suffix(f"{self.file_path.suffix}.{os.getpid()}.tmp")
        try:
            df_to_save.write_ipc(temp_file_path, compression="lz4")
            shutil.move(str(temp_file_path), str(self.file_path))
        except Exception as e:
            if temp_file_path.exists():
                try:
                    temp_file_path.unlink(missing_ok=True)
                except OSError:
                    if temp_file_path.exists():
                        os.remove(temp_file_path)
            raise OSError(f"Failed to save data to {self.file_path}: {e}") from e
        finally:
            if temp_file_path.exists():
                try:
                    temp_file_path.unlink(missing_ok=True)
                except OSError:
                    if temp_file_path.exists():
                        os.remove(temp_file_path)

    def get_data(self) -> pl.DataFrame:
        """Returns the full DataFrame from memory."""
        with self._lock:
            return self._load_data()

    def update_data(self, new_data: pl.DataFrame) -> None:
        """
        Updates the stored data with new data points.

        This method combines the current data with the new data, sorts by the
        datetime column, removes duplicates (keeping the last entry), truncates
        to max_rows, and saves the result atomically.
        """
        if new_data is None or new_data.is_empty():
            return

        with (self._lock):
            try:
                conformed_new_data = self._conform_df_to_schema(new_data, "new_data_input")
            except Exception as e:
                raise ValueError(f"Fatal schema error in new_data for {self.file_path}: {e}") from e

            current_data = self._load_data()
            try:
                combined_data = pl.concat(items=[current_data, conformed_new_data],
                                      how="vertical_relaxed") if not current_data.is_empty() else conformed_new_data
            except Exception as e:
                print(e)
                print(self.file_path)
                print(current_data.schema)
                print(conformed_new_data.schema)
                has_struct = any(isinstance(dtype, pl.Struct) for dtype in current_data.schema.values())
                print(has_struct)
                has_struct = any(isinstance(dtype, pl.Struct) for dtype in conformed_new_data.schema.values())
                print(has_struct)

            if self.datetime_col not in combined_data.columns:
                raise ValueError(f"Internal error: Datetime column '{self.datetime_col}' not found in combined data.")

            if combined_data.get_column(self.datetime_col).null_count() != len(combined_data):
                combined_data = combined_data.sort(by=self.datetime_col).unique(subset=[self.datetime_col], keep="last",
                                                                                maintain_order=False).sort(
                    by=self.datetime_col)
            else:
                print(
                    f"Warning: Datetime column '{self.datetime_col}' in {self.file_path} is all nulls. Cannot de-duplicate by time.")

            if len(combined_data) > self.max_rows:
                combined_data = combined_data.tail(self.max_rows)

            self._save_data(combined_data)

    def get_shape(self) -> tuple[int, int]:
        """Returns the shape of the stored DataFrame."""
        with self._lock:
            if not self.file_path.exists() or self.file_path.stat().st_size == 0:
                return (0, len(self.schema))
            df = self._load_data()
            return df.shape

    def get_latest_rows(self, n: int) -> pl.DataFrame:
        """Returns the n most recent rows."""
        if n <= 0:
            return pl.DataFrame(data={}, schema=self.schema)
        with self._lock:
            df = self._load_data()
            return df.tail(n)

    def get_oldest_rows(self, n: int) -> pl.DataFrame:
        """Returns the n oldest rows."""
        if n <= 0:
            return pl.DataFrame(data={}, schema=self.schema)
        with self._lock:
            df = self._load_data()
            return df.head(n)

    def clear(self, n: int | None = None) -> None:
        """
        Clears or truncates data from the FactorMemory.
        Behavior depends on the instance's mode.

        - In BACKTEST mode: Always deletes the file to ensure a clean slate for the next run.
        - In LIVE mode:
            - If n is None or 0: Clears data but keeps an empty, initialized file.
            - If n > 0: Truncates data, keeping the `n` most recent rows.
        """
        with self._lock:
            # --- BACKTEST MODE ---
            # In backtesting, we always want a clean slate. The most reliable
            # way to ensure this is to delete the cache file.
            if self.mode == FactorMode.BACKTEST:
                try:
                    if self.file_path.exists():
                        self.file_path.unlink()
                        print(f"Factor memory file deleted (Backtest Mode): {self.file_path}")
                except Exception as e:
                    print(f"Error during file deletion for {self.file_path} (Backtest Mode): {e}")
                return

            # --- LIVE MODE ---
            # In live trading, we prioritize data preservation.
            if n is not None and n > 0:
                # Truncate to the n most recent rows
                current_data = self._load_data()
                if n < len(current_data):
                    truncated_data = current_data.tail(n)
                    self._save_data(truncated_data)
                    print(f"Factor memory truncated to {n} rows (Live Mode): {self.file_path}")
                else:
                    print(f"Truncation skipped: request ({n}) >= current rows ({len(current_data)}). No change made.")
            else:
                # Clear all data but re-initialize the file with an empty schema.
                try:
                    empty_df = pl.DataFrame(data={}, schema=self.schema)
                    self._save_data(empty_df)
                    print(f"Factor memory cleared and re-initialized (Live Mode): {self.file_path}")
                except Exception as e:
                    print(f"Error during clear/re-initialization for {self.file_path} (Live Mode): {e}")

    @property
    def is_empty(self) -> bool:
        """Checks if the stored DataFrame is empty."""
        return self.get_shape()[0] == 0

    def __repr__(self) -> str:
        """Returns a string representation of the FactorMemory instance."""
        return (
            f"FactorMemory(file_path='{self.file_path}', max_rows={self.max_rows}, "
            f"mode={self.mode.name}, schema_cols={list(self.schema.keys())})"
        )


@dataclass
class MemoryData(pl.DataFrame):
    """
    MemoryData is a subclass of polars DataFrame that is used to store
    data in memory for fast access and manipulation.
    It is used for storing tick data, bar data, and factor data.
    """

    _max_rows: int = field(default=None, init=False)
    interval: Interval = field(default=None, init=False)
    time_range: TimeRange = field(default=None, init=False)
    datetime_col: str = field(default="datetime", init=False)

    def __init__(self,
                 data=None,
                 schema=None,
                 schema_overrides=None,
                 strict=True,
                 orient=None,
                 infer_schema_length=100,
                 nan_to_null=False,
                 max_rows: int = None,
                 interval: Interval = None,
                 datetime_col: str = "datetime",
                 **kwargs):
        super().__init__(data=data,
                         schema=schema,
                         schema_overrides=schema_overrides,
                         strict=strict,
                         orient=orient,
                         infer_schema_length=infer_schema_length,
                         nan_to_null=nan_to_null)

        self._max_rows = max_rows if max_rows is not None else 1000  # Default max rows to 1k
        self.interval = interval
        self.datetime_col = datetime_col
        self.sort(by=self.datetime_col, descending=False, nulls_last=False)
        self.time_range: TimeRange = TimeRange(start=self[0, "datetime"],
                                               end=self[-1, "datetime"],
                                               interval=interval)  # Used to store the time range of the data

    def vstack_truncated(self, other: Union[Self, pl.DataFrame]) -> None:
        """
        Append data to the MemoryData DataFrame.
        If the number of rows exceeds max_rows, it will drop the oldest rows.
        """
        assert self.time_range.overlaps(TimeRange(start=other[0, "datetime"],
                                                  end=other[-1, "datetime"], interval=self.interval))
        self.vstack(other, in_place=True)
        if self._max_rows and self.height > self._max_rows:
            self._df = self._df.tail(self._max_rows)
