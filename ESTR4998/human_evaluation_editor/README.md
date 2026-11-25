# Human evaluation editor

Try the six interfaces used in the study with an invented museum exhibit task. The grouping panel asks whether two designs express the same central idea. The other panels support pair equivalence, four quality scores, rubric calibration, manual inspection and fixture checks. Every human answer starts blank. In the inspection panel the score is an illustration supplied by the demo.

## Run the example

From this directory on Linux or macOS, run the following command with Python 3.10 or later. The editor uses the Python standard library.

```sh
python3 demo.py
```

Open [Rater A’s local editor](http://127.0.0.1:8081) and sign in with the temporary password printed in your terminal. Assign group numbers to the three responses, record prior exposure and select **Mark complete**. Reload to see the saved assignments or select **Download my CSV** to keep a copy.

For a second independent sheet, start another terminal in the same directory and run `python3 demo.py --rater B`. Open [Rater B’s local editor](http://127.0.0.1:8082) and use the password from the second terminal. Each running instance has its own temporary sheets and backups. Press Ctrl+C to end a session and remove its temporary files. The `--port` argument lets you choose another local port.

## Verify the release

```sh
python3 -m unittest -v test_demo
```

The tests exercise sign-in, all six panels, saving and reloading a completed grouping, and its CSV export. They run through the real HTTP handler using in-memory byte streams. A second test repeats these steps through a loopback socket when the host permits it. The grouping schema also checks missing assignments, unknown and duplicate response IDs, prior exposure and notes for unresolved judgments.

The interface and server derive from thesis commit `674b954`. This release uses anonymous labels, binds to loopback and exposes all six panels in the navigation. The grouping schema retains the row checks used by the editor. The demo builds its examples locally and includes no original participant materials.
