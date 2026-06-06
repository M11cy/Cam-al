# Local Mac Camera Workplace Monitor

This prototype connects to your Mac camera and tracks:

- whether an employee is present in the selected work zone;
- whether the employee has been absent longer than the configured limit;
- whether a phone is visible near the employee;
- whether a phone has remained visible for more than 60 seconds.

The app uses stricter confirmation rules for phones to reduce false positives
from similar objects such as combs, vapes, cans, and small packages. Gray boxes
are ignored phone candidates. Red boxes are confirmed phones that start the
alert timer.

## Run

```bash
./run.sh
```

If the camera does not open, try another camera index:

```bash
./run.sh --camera 1
```

## Controls

- `r` - select the work zone with the mouse;
- `s` - save the current zone to `config.json`;
- `q` or `Esc` - quit.

On macOS, you may need to allow Terminal/Python camera access in
`System Settings -> Privacy & Security -> Camera`.

## Configuration

After pressing `s`, the app writes `config.json`. You can edit:

- `absence_seconds` - seconds before employee absence alert;
- `phone_alert_seconds` - seconds before phone alert, default `60`;
- `confidence` - person detection confidence;
- `phone_confidence` - phone detection confidence.
- `phone_min_area_ratio` / `phone_max_area_ratio` - accepted phone box size;
- `phone_min_aspect_ratio` / `phone_max_aspect_ratio` - accepted phone box shape;
- `phone_person_zone_scale` - how close the phone must be to the employee.
