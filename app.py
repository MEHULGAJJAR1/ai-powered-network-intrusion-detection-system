#!/usr/bin/env python3
"""Local entry point. Production containers run the same factory through Gunicorn."""
import os

from nids.api.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False, threaded=True)
