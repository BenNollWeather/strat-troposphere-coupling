# Polar vortex coupling

Day-15 ensemble-mean height from the ECMWF EPS and AIFS-ENS, and whether a northern stratospheric warming is coupling downward.

Each model gets two stills. The northern map looks down on the North Pole. The southern map is drawn under it on the page: it stays centered on the South Pole, with New Zealand (174°E) at the top. Both shade the height anomaly in standard deviations from the ERA5 1991–2020 average and draw the actual height as contours, at 10, 100, 300 and 500 hPa.

The scoreboard rates only the Northern Hemisphere. A headline such as “No coupling predicted on Nov. 13” uses the forecast valid date.

## Schedule

GitHub Actions does not use its own cron. AWS EventBridge starts the workflow when the day-15 files usually arrive.

| Model | Cycles | Typical arrival | EventBridge (UTC) |
|---|---|---|---|
| AIFS-ENS | 00, 06, 12, 18 | about 7 h 45 m after the cycle | 01:55, 07:55, 13:55, 19:55 |
| EPS (IFS ensemble) | 00, 12 | about 8 h 25 m after the cycle | 08:35, 20:35 |

The job checks the open-data index and waits up to 50 minutes if that cycle is a few minutes late. If it still is not there, the previous complete cycle is used. A cycle that is already published, including its southern map, is not downloaded again.

Deploy `aws/eventbridge-github-dispatch.yaml` twice, reusing the shared GitHub connection. One stack uses model `aifs-ens` and `cron(55 1,7,13,19 * * ? *)`. The other uses model `ifs-ens` and `cron(35 8,20 * * ? *)`.

## Forecast fields are not stored

ECMWF GRIB messages and ARCO-ERA5 chunks are read into memory. They are not written to disk, uploaded, or committed. Coastlines and the renders sit in a temporary directory outside the home folder. That directory is deleted when the process exits, including when the run fails. The only files published are the two PNGs per model, `latest.json`, a short diagnostics text file, the page, and a status JSON.

## Page

https://bennollweather.s3.amazonaws.com/climate_imagery/strat_coupling/index.html

The typeface is Libre Franklin, SIL Open Font License 1.1 (`assets/fonts/OFL.txt`). The page loads the same family from the bunny font stylesheet used by the other graphics pages. The maps embed the files in `assets/fonts/`.

## Run

```bash
bash bin/run
```

`MODEL=aifs-ens` or `MODEL=ifs-ens` limits the run to one model. `UPLOAD_TO_S3=false` draws without uploading. The repository secret `AWS_ROLE_ARN` is the role GitHub Actions assumes to write to the public bucket. No AWS profile is set in this repo.
