# dartcv

WIP: Dart computer vision framework enabling a smart phone to count your darts score automatically.

## Pictures on Google Drive (uv + DVC)

Pictures in `data/images/` are managed by DVC, not committed directly to Git.
Git tracks `data/images.dvc`, which identifies the version of the entire image
folder. DVC stores the image contents on Google Drive using the configured
default remote, `gdrive-remote`.

Run the following commands from the repository root.

### Initial setup and authentication

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and sync
the project environment (this also installs DVC with Google Drive support):

```sh
uv sync
```

Ask a project maintainer for access to the Google Drive remote folder and the
OAuth client credentials. Configure credentials locally; replace the
placeholders below with the supplied values:

```sh
uv run dvc remote modify --local gdrive-remote gdrive_client_id '<CLIENT_ID>'
uv run dvc remote modify --local gdrive-remote gdrive_client_secret '<CLIENT_SECRET>'
```

The `--local` flag saves these values in `.dvc/config.local`, which is ignored
by Git. Never commit credentials or authentication tokens, or paste them into
issues or chat. Commands containing actual secrets may also be saved in shell
history; handle that history as sensitive.

The first pull or push may open a browser for Google authorization. Use an
account with access to the remote folder and keep the terminal command running
until authentication finishes.

If Safari blocks the redirect to `http://localhost:8080` because HTTPS-Only mode
is enabled, open the full callback URL in another browser on the same Mac while
DVC is still running. Alternatively, temporarily disable HTTPS-Only mode and
reload, then re-enable it afterward. Do not change the callback URL to HTTPS.
If DVC has stopped, restart the operation to begin a fresh login. Treat callback
URLs as sensitive because they contain temporary authorization codes.

### Download the latest pictures

```sh
git pull
uv run dvc pull
```

`git pull` fetches the latest DVC metadata; `uv run dvc pull` downloads and
checks out the corresponding pictures into `data/images/`. Preserve any local
image edits first; do not force a pull over changes you want to keep.

### Upload added, edited, or deleted pictures

After changing files in `data/images/`:

```sh
# Inspect local changes relative to the recorded image version.
uv run dvc status

# Record the entire folder, including additions, edits, and deletions.
uv run dvc add data/images

# Upload the recorded image contents to Google Drive.
uv run dvc push

# Publish the metadata so others can download the same image version.
git add data/images.dvc data/.gitignore
git commit -m "Update image dataset"
git push
```

Use `uv run dvc add data/images`, not `uv run dvc add data/images/*`: the folder
is already tracked as a single dataset. A push alone does not record new image
changes; run `dvc add` first. Upload with DVC before pushing the Git commit so
the referenced data is available to collaborators. DVC retains older cached
versions; deleting a local picture does not purge its historical contents from
Google Drive.

### Check synchronization

```sh
# Compare local pictures with the version recorded in Git/DVC metadata.
uv run dvc status

# Compare the local DVC cache with Google Drive.
uv run dvc status -c
```

These checks serve different purposes: a synchronized remote cache does not
mean unrecorded local image changes have been uploaded. For those changes,
repeat the add, DVC push, Git commit, and Git push workflow above.

