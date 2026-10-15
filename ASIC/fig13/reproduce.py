from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parent
if __name__ == "__main__":
    subprocess.run([sys.executable, str(ROOT / "scripts/plot_dual_es.py"), str(ROOT / "data/raw"), "--output-prefix", str(ROOT / "figure/fig13_es1_es2_cross_correlation")], check=True)
