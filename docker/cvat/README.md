# Local CVAT

Uses the pinned `thirdparty/cvat` checkout and official `v2.84.0` images.
Requires Docker Compose 2.24.4+ and a running Docker engine. On Apple Silicon,
enable Rosetta in OrbStack for the default AMD64 images.

From the repository root, prepare the settings and start CVAT:

```bash
git submodule update --init -- thirdparty/cvat
cp -n docker/cvat/.env.example docker/cvat/.env

./docker/cvat/cvat config --quiet                 # Validate settings
./docker/cvat/cvat up -d                          # Download images and start
./docker/cvat/cvat ps                             # Check containers
./docker/cvat/cvat logs --tail=50 cvat_server      # Inspect startup
```

After startup completes, check readiness and create an administrator:

```bash
./docker/cvat/cvat exec cvat_server python manage.py health_check
./docker/cvat/cvat exec cvat_server python manage.py createsuperuser
```

Open <http://localhost:8080> in Chrome (or the port set in `.env`).
Stop with `./docker/cvat/cvat down`; named volumes retain annotation data.
Avoid `down -v`. The executable also works from other directories via its path.

Save project/task backups from CVAT's UI under `outputs/cvat/backups/` and COCO
dataset exports with images under `data/cvat/exports/`. For complete server
backups, see the [CVAT backup guide](https://docs.cvat.ai/docs/administration/community/advanced/backup_guide/).
Custom CVAT source changes require rebuilding the images.
