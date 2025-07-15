import requests
import matplotlib.pyplot as plt
from pathlib import Path
def download(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        # with ensures automatic closing after write
        with open(filename, "wb") as f: # create/open file; "wb": w = write file, b = binary mode(req for non-text files to prevent corruption)
            f.write(response.content)


def savePlot(path):
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return 

def checkIfPlotExists(path: str) -> bool:
    return Path(path).exists()