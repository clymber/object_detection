# Local CVAT

Uses the pinned `thirdparty/cvat` checkout and official `v2.84.0` images.
Requires Docker Compose 2.24.4+ and a running Docker engine. On Apple Silicon,
enable Rosetta in OrbStack for the default AMD64 images.

From the repository root, prepare the settings and define a shell helper:

```bash
git submodule update --init -- thirdparty/cvat
cp -n docker/cvat/.env.example docker/cvat/.env

# Run CVAT Compose commands with the local settings and override.
cvat() {
  docker compose --env-file docker/cvat/.env \
    -f thirdparty/cvat/docker-compose.yml \
    -f docker/cvat/compose.local.yml "$@"
}

cvat config --quiet                 # Validate settings
cvat up -d                         # Download images and start
cvat ps                            # Check containers
cvat logs --tail=50 cvat_server     # Inspect startup
```

After startup completes, check readiness and create an administrator:

```bash
cvat exec cvat_server python manage.py health_check
cvat exec cvat_server python manage.py createsuperuser
```

Open <http://localhost:8080> in Chrome (or the port set in `.env`).
Stop with `cvat down`; named volumes retain annotation data. Avoid `down -v`.
The shell helper must be defined again in each new terminal.

Save project/task backups from CVAT's UI under `outputs/cvat/backups/` and COCO
dataset exports with images under `data/cvat/exports/`. For complete server
backups, see the [CVAT backup guide](https://docs.cvat.ai/docs/administration/community/advanced/backup_guide/).
Custom CVAT source changes require rebuilding the images.
