import os


def to_csv(df, path, **kwargs):
    if path.startswith("s3://"):
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
        storage_options = {"endpoint_url": endpoint_url} if endpoint_url else {}
        df.to_csv(path, index=False, storage_options=storage_options, **kwargs)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False, **kwargs)
