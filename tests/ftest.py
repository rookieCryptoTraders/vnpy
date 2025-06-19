import polars as pl
import polars_talib as plta

df = pl.DataFrame({
    "Close": [10, 12, 15, 13, 16, 18, 20, 22, 25, 23]
})

df = df.with_columns(
    plta.bbands(pl.col("Close"))
)

print(df)
