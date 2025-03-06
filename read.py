import marimo

__generated_with = "0.11.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import av
    import sys
    import PIL.Image
    import numpy as np
    import matplotlib.pyplot as plt

    from tqdm import tqdm
    return PIL, av, mo, np, plt, sys, tqdm


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
