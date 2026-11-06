"""Publish verified models privately and issue expiring pilot download URLs.

Uses the AWS CLI credential chain on the developer machine, never in the APK.
"""

import argparse
import datetime
import hashlib
import json
import subprocess
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--profile", default="strataiforge")
    parser.add_argument("--region", default="ap-south-1")
    parser.add_argument("--prefix", default="mobile/v1")
    parser.add_argument("--manifest-key", default="mobile/latest.json")
    parser.add_argument("--preserve-app-config", action="store_true",
                        help="Publish a release without changing the app's bundled pilot config")
    parser.add_argument("--catalog", type=Path, default=ROOT / "mobile/model-catalog.json")
    parser.add_argument("--expires-in", type=int, default=604800)
    parser.add_argument("--create-bucket", action="store_true")
    parser.add_argument("--urls-only", action="store_true")
    args = parser.parse_args()
    if not 60 <= args.expires_in <= 604800:
        parser.error("S3 presigned URLs must expire in 60..604800 seconds")
    if args.create_bucket and args.urls_only:
        parser.error("--create-bucket cannot be used with --urls-only")

    def aws(*command):
        result = subprocess.run(
            ["aws", "--profile", args.profile, "--region", args.region, "--no-cli-pager", *command],
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip()

    catalog = json.loads(args.catalog.read_text())
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", catalog["version"]):
        raise ValueError("Use a safe, unique model-pack version identifier")
    if args.create_bucket:
        options = [] if args.region == "us-east-1" else [
            "--create-bucket-configuration", f"LocationConstraint={args.region}"]
        aws("s3api", "create-bucket", "--bucket", args.bucket, *options)
        aws("s3api", "put-public-access-block", "--bucket", args.bucket,
            "--public-access-block-configuration",
            "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true")
        aws("s3api", "put-bucket-encryption", "--bucket", args.bucket,
            "--server-side-encryption-configuration",
            json.dumps({"Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]}))
    urls = {}
    for artifact in catalog["artifacts"]:
        path = ROOT / "mobile-assets" / artifact["filename"]
        extension = "tflite" if artifact["id"] == "classifier" else "gguf"
        key = f"{args.prefix}/artifacts/{artifact['sha256']}.{extension}"
        if not args.urls_only:
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            if checksum != artifact["sha256"] or path.stat().st_size != artifact["bytes"]:
                raise ValueError(f"Local artifact changed: {artifact['id']}")
            aws("s3", "cp", str(path), f"s3://{args.bucket}/{key}",
                "--sse", "AES256", "--metadata", f"sha256={checksum}", "--only-show-errors")
        remote = json.loads(aws("s3api", "head-object", "--bucket", args.bucket, "--key", key))
        if remote["ContentLength"] != artifact["bytes"] or remote.get("Metadata", {}).get("sha256") != artifact["sha256"]:
            raise ValueError(f"S3 artifact does not match catalog: {artifact['id']}")
        urls[artifact["id"]] = aws("s3", "presign", f"s3://{args.bucket}/{key}",
                                   "--expires-in", str(args.expires_in))
    if not args.urls_only:
        for path in [args.catalog, ROOT / "mobile-assets/conversion-report.json",
                     ROOT / "mobile-assets/QWEN-LICENSE.txt"]:
            aws("s3", "cp", str(path), f"s3://{args.bucket}/{args.prefix}/{path.name}",
                "--sse", "AES256", "--only-show-errors")
    expires = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=args.expires_in)
    config = {"urls": urls, "expiresAt": expires.isoformat(), "bucket": args.bucket,
              "catalogVersion": catalog["version"]}
    manifest = {"catalog": catalog, "downloads": config}
    # Upload the mutable pointer last; artifacts use content-addressed keys.
    manifest_path = ROOT / "mobile-assets/latest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_path.chmod(0o600)
    aws("s3", "cp", str(manifest_path), f"s3://{args.bucket}/{args.manifest_key}",
        "--sse", "AES256", "--content-type", "application/json",
        "--cache-control", "no-store", "--only-show-errors")
    config["manifestUrl"] = aws("s3", "presign", f"s3://{args.bucket}/{args.manifest_key}",
                               "--expires-in", str(args.expires_in))
    destination = ROOT / "mobile/download-config.json"
    if not args.preserve_app_config:
        destination.write_text(json.dumps(config, indent=2) + "\n")
        destination.chmod(0o600)
    print(f"Private models verified in s3://{args.bucket}/{args.prefix}/")
    print(f"Published latest manifest; artifact links expire no later than {expires.isoformat()}.")
    if not args.preserve_app_config:
        print("Pilot manifest access link written to local app configuration.")
    print("Temporary AWS credentials can make links expire earlier. Rebuild the pilot APK after refreshing.")


if __name__ == "__main__":
    main()
