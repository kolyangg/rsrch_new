# Dropbox artifact utility

Copied from `/home/kolyangg/Other/emdr/tools/dropbox/dropbox_tool.py` with the
default project folder changed to `/rsrch_new`. Uses only Python's standard
library, OAuth refresh tokens, verified Dropbox content hashes and temporary
download links.

The copied credentials use the Dropbox app folder `Apps/temp`: API path
`/rsrch_new` is visible as `Apps/temp/rsrch_new` in Dropbox. This mapping was
verified by matching the existing EMDR folder ID through both API views.
Do not prepend `/Apps/temp` to paths passed to this helper.

Credentials live in ignored `.env`, mode 0600: `DROPBOX_APP_KEY`,
`DROPBOX_APP_SECRET`, `DROPBOX_REFRESH_TOKEN`. The configured
`DROPBOX_PROJECT_FOLDER` is `/rsrch_new`. The helper reads `.env.local` before
`.env` and preserves process environment overrides. Never print or upload
credential values.

```bash
python3 tools/dropbox/dropbox_tool.py upload \
  reports/261001_flux4b_branched_attention/flux4b_branched_attention_architecture.pdf \
  --date 2026-10-01
python3 tools/dropbox/dropbox_tool.py download 2026-10-01/flux4b_branched_attention_architecture.pdf
```

Uploads go to `/rsrch_new/YYYY-MM-DD/<filename>` and overwrite a same-day file
with the same name. Downloads default to ignored `dropbox/`. Transfer only
requested artifacts. Temporary direct-download links expire after about four
hours; uploads do not create permanent public shared links.
