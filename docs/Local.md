# Local Use

To generate contactomes and connectomes locally, you can use the following commands. This will not deploy any cloud resources (no SQS, no Lambdas).

## Provision the queue

```bash
uv run python3 local_manage.py contactome generate --graph-id test0 --segmentation-channel s3://cvdb-bossdb-boss/martinez2025/zebrafish/shuffle_1_checkpoint_5000
```
