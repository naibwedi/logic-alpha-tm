# Licensed data protocol

## Tiingo Starter: preliminary long-history benchmark

The free Tiingo Starter path provides sufficient request and symbol limits for
SPY, QQQ, IWM, and TLT. Tiingo data are licensed for internal use; never commit,
redistribute, or embed downloaded rows in public artifacts.

The downloader records adjusted and raw closes, dividend/split fields, a download
timestamp, checksums, and a conservative 20:00 America/New_York availability
assumption. Tiingo may correct historical records, so this is current-vintage
history rather than a historical revision archive.

```powershell
$env:TIINGO_API_TOKEN = "your-local-token"
python -m logic_alpha_tm.cli download-tiingo --start 2005-01-01 --end 2025-12-31
python -m logic_alpha_tm.cli benchmark --csv data/raw/tiingo-prices.csv --spec experiments/tiingo-v0.2.json --phase development --output results/tiingo-development-v0.2
```

Do not open the locked holdout until development choices are frozen.

### Private Colab execution

`notebooks/tiingo_gpu_colab.ipynb` runs the development benchmark on a Colab
NVIDIA runtime without storing the Tiingo token. Upload only the adjusted price
and availability files into the temporary private runtime. Download the result
ZIP, then disconnect and delete the runtime. The notebook, repository, and any
shared drive must never contain the licensed CSV files.

## Massive: stricter licensed-data path

The real-data path uses [Massive U.S. Stocks daily aggregate bars](https://massive.com/docs/rest/stocks/aggregates/custom-bars).
Access requires
a Massive account, an API key, and a plan whose license covers the intended use.
The downloader sends the key in an `Authorization: Bearer` header and never writes
it to disk.

## Point-in-time policy

- Download `adjusted=false` bars so later corporate-action adjustments are not
  silently applied to earlier observations.
- Keep downloaded files under `data/raw/`; Git ignores this vendor-controlled data.
- Record `observation_at`, `available_at`, source, and revision metadata alongside
  prices. The adapter records 16:15 America/New_York on the observation date as a
  conservative post-close assumption; verify it against the latency of your plan.
- Make decisions only when `available_at <= decision_at`.
- Record the download date, vendor plan, API response status, and experiment commit.

Price-only unadjusted bars omit dividends and therefore are not a total-return
series. Before making financial claims, add point-in-time corporate actions or a
licensed total-return source and test survivorship, delisting, and revision bias.

## Reproduction

```powershell
pip install -e ".[tm]"
$env:MASSIVE_API_KEY = "your-local-key"
python -m logic_alpha_tm.cli download-massive --start 2005-01-01 --end 2025-12-31 --vendor-plan "your-plan-label"
python -m logic_alpha_tm.cli run --csv data/raw/massive-prices.csv --model bernoulli --output results/real-bernoulli
python -m logic_alpha_tm.cli run --csv data/raw/massive-prices.csv --model tmu --output results/real-tmu
```

The download also creates `massive-prices.available-at.csv` and
`massive-prices.manifest.json`. The frozen benchmark refuses to run without the
availability file and fingerprints both files in its output manifest.

See Massive's [authentication guidance](https://massive.com/docs/rest/quickstart)
and [stocks documentation](https://massive.com/docs/rest/stocks) for current API
and entitlement details. Do not paste credentials into issues, logs, reports, or chat. The repository's
synthetic demo stays in CI because licensed vendor data cannot be redistributed.
