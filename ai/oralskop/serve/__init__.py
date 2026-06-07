"""FastAPI serving layer for the trained torchseg (semantic-segmentation) model.

`model.py` loads an OralSkop torchseg checkpoint and runs inference on arbitrary
RGB photos (replicating the training-time preprocessing). `app.py` wraps it in a
FastAPI app (JSON detections + an annotated-overlay PNG + a browser upload form).
`notebook.py` runs that app from inside a Jupyter cell and opens a public ngrok
URL so a remote friend can call it. See `notebooks/serve_api.ipynb`.
"""
