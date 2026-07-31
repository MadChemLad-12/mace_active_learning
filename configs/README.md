# Config structure

```
configs/
  constants.py                     # stable chemistry/infra facts, rarely edited
  neb_model_compare_config.py      # config for the (non-pipeline) comparison tool
  round_configs/
    round1_active_pipeline.py      # per-round tunables for src/active_pipeline.py
    round1_check_residual.py       # per-round tunables for src/check_residual.py
    round1_neb_geo_run.py          # per-round tunables for neb_geo_run.py
```

## Round index
- R6: <<one-line summary of what changed this round, fill in as you go>>

## How to start a new round
```bash
cd configs/round_configs
cp round1_active_pipeline.py   round2_active_pipeline.py
cp round1_check_residual.py    round2_check_residual.py
cp round1_neb_geo_run.py       round2_neb_geo_run.py
# edit the new files, then:
git diff round1_active_pipeline.py round2_active_pipeline.py
```
