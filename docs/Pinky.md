## Pinky100

```bash
AWS_PROFILE=bossdb uv run local_manage.py \
    --sqlite-db-path pinkycontactome.db --shard-sqlite \
    --mip 64,64,40 \
    contactome generate \
    --graph-id pinky100 \
    --segmentation-channel precomputed://s3://bossdb-open-data/iarpa_microns/pinky/seg
```
