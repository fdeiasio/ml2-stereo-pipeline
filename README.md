# Magic Leap Stereo Pipeline — Server

Python host-side application for a Magic Leap 2 augmented reality setup. The headset streams
the raw feed from its three world cameras over a USB (adb) TCP link; this server reconstructs
depth from the stereo pair, fits a known 3D model into the resulting point cloud, and streams
back the estimated 6-DoF pose of that model.

It is the counterpart to the Unity client: [magicleapclient](../magicleapclient).

## Pipeline

```
  Magic Leap 2                                    host (this repo)
  ┌───────────┐   TCP :9000  WCAM packet    ┌──────────────────────────────┐
  │ 3 world   │ ──────────────────────────► │ parse  -> decode             │
  │ cameras   │   left / center / right     │        -> rectify            │
  │ 1016x1016 │   30 fps, uncompressed      │        -> disparity (BM)     │
  └───────────┘                             │        -> point cloud        │
                                            │        -> ICP model fit      │
                                            └───────────────┬──────────────┘
                                                            │  UDP :12345
                                                            ▼  "x,y,z,qx,qy,qz,qw"
                                                     downstream application
```

Every stage is optional and controlled from `magicleap_server/config.py`. `PREVIEW_MODE`
selects what is shown locally (`raw`, `rect`, `disp`, `view3d`, `model` or `none`) and
`STREAM_TO_REMOTE` decides whether the rest of the pipeline still runs and is forwarded. The
pipeline degrades instead of crashing: if a rectification map or the model fails to load, the
server falls back to the previous stage and turns streaming off.

## Requirements

- Python 3.12. Open3D does not support 3.13+, so a newer interpreter will fail at install time.
- `opencv-python`, `numpy`, `open3d` (see `requirements.txt`)
- A Magic Leap 2 with the client application installed, and `adb` on the PATH

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

adb reverse tcp:9000 tcp:9000    # forward the device's loopback to this machine
python main.py
```

`adb reverse` is what makes the device able to reach the host on `127.0.0.1:9000` over the
USB cable. It is reset whenever the device reconnects, so re-run it if the client stops
connecting.

To watch the pose stream without the headset, run `python streaming_test.py` in a second
terminal. It binds UDP 12345 and prints every packet it receives; use it as a starting point
for anything that consumes the pose.

## Configuration

Everything lives in `magicleap_server/config.py`, grouped by pipeline stage:

| Setting | Meaning |
| --- | --- |
| `CAMERA_PAIR` | `left-center` (narrow baseline, close range) or `center-right` (wider baseline, better depth precision). |
| `PREVIEW_MODE` | Which stage to render locally. `none` computes everything without a window. |
| `STREAM_TO_REMOTE` | Compute and stream the full pipeline even when the preview stops early. |
| `HOST` / `PORT` | Listen address for the device connection. Default `0.0.0.0:9000`. |
| `DEFAULT_NUM_DISPARITIES` | Disparity search range. Default `128`, tuned for arm's-length working distance. |
| `DEFAULT_BLOCK_SIZE` | StereoBM correlation window. Trades noise against detail. |
| `MAX_POINT_COUNT` | Upper bound on points kept per frame, for latency. |
| `MIN_VALID_DISPARITY` | Floor used to reject far-field noise. |
| `MODEL_PATH` / `MODEL_SCALE_FACTOR` | Reference mesh and the hand-tuned scale applied to it. |
| `DEFAULT_POSITION` / `DEFAULT_ORIENTATION` | ICP initial guess, used for the first frame and after a rejected fit. |
| `ICP_MIN_FITNESS` | Fitness below which an ICP result is discarded and the previous pose is kept. |
| `STREAMING_PORT` | UDP port for the outgoing pose stream. Default `12345`. |

## Wire protocol

**Device to server (TCP).** One packet per frame batch. All integers little-endian.

| Field | Size | Value |
| --- | --- | --- |
| Magic header | 4 B | ASCII `WCAM` |
| Frame block 1..3 | — | One block per camera |
| ↳ metadata length | 4 B int32 | |
| ↳ metadata | n B | UTF-8 JSON: `cameraId`, `frameType`, `frameNumber`, `width`, `height`, `stride`, `bytesPerPixel`, `timestampNs` |
| ↳ image length | 4 B int32 | |
| ↳ image | n B | Uncompressed 8-bit grayscale, `width * height` bytes |

A missing camera is encoded as a zero-length metadata block followed by a zero-length image.
Frames are matched by `cameraId` rather than by position, so block order does not matter.

**Server to downstream (UDP).** A single ASCII line per frame:
`x,y,z,qx,qy,qz,qw`, each formatted to 8 decimal places. No header, no framing.

## Calibration

The pipeline needs intrinsics, stereo extrinsics and rectification maps. Because these are the
output of a physical procedure, the results are committed under `data/` so a clone runs
without a headset; see [data/README.md](data/README.md) for what is tracked and why.

```bash
python calibrate.py          # mono intrinsics       -> data/calibration.json
python stereo_calibrate.py   # stereo + rectification -> data/stereo_calibration_*.json + .npz
```

Both scripts read a 7x6 checkerboard with 25 mm squares, and both can show you the detected
corners so you can keep or discard each capture (`y` / `d`).

## Repository layout

```
main.py                        entry point
calibrate.py                   mono camera calibration
stereo_calibrate.py            stereo calibration and rectification map generation
streaming_test.py              reference UDP consumer for the pose stream

magicleap_server/
  server.py                    TCP accept loop, one thread per client
  client.py                    the pipeline for a single connection
  parser.py                    WCAM packet framing
  decoder.py                   raw bytes -> cv2 image
  rectifier.py                 rectification map loading and remap
  disparity.py                 StereoBM matching
  point_cloud.py               Q-matrix reprojection and depth filtering
  pose_estimation.py           Open3D ICP against the reference mesh
  previewer.py                 2D preview windows, 's' saves frames
  visualize_3d.py              Open3D point cloud viewer
  model_preview.py             mesh overlay preview
  utils.py                     geometry and parsing helpers
  config.py                    every tunable, documented inline

data/                          calibration results and reference mesh
```

## Performance

The workload is dominated by `reprojectImageTo3D` and ICP, in that order; both are CPU-bound
and both are the place to start if this needs to be faster. StereoBM was chosen over
StereoSGBM deliberately: SGBM gives cleaner maps but was too slow on CPU to hold the frame
rate, and offloading it to the GPU was left as future work.

The client paces the stream at 30 fps. Each packet carries three 1016x1016 grayscale frames,
about 3.1 MB, so the link sustains roughly 93 MB/s. That is why the connection is a USB cable
with `adb reverse` rather than WiFi.

## Limitations

- The pose stream is one-way. `adb reverse` cannot forward UDP, so the device cannot receive
  the pose over the same link; the client's pose path is left unimplemented. Anything that
  consumes the pose runs on the host.
- `MODEL_SCALE_FACTOR = 2/3` is empirical. The mesh and the reconstructed point cloud do not
  agree on scale, and the likely cause has not been tracked down.
- The three world cameras are assumed to share one set of intrinsics. `calibrate.py` pools
  captures from all of them.
- The pose estimate is expressed in the rectified camera frame. Anchoring it to a world
  reference (an ArUco marker) was designed but not built.
- No tests.

## License

MIT — see [LICENSE](LICENSE).
