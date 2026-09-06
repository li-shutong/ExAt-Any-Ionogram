#!/usr/bin/env python3
"""Convenience entry point for the ionogram-scaling skill.

Usage:
    python run.py --image ionogram.png
    python run.py --batch data/
    python run.py --batch data/ --threshold 0.90 --max-retries 3
"""

from ionogram_scaling.main import main

if __name__ == "__main__":
    main()
