#This file contains utility functions, primarily for GitHub integration.

import subprocess
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def commit_and_push():
    """Commit and push changes to GitHub."""
    try:
        subprocess.run(["git", "add", "."], cwd=os.getcwd(), check=True, capture_output=True, text=True)
        result = subprocess.run(["git", "commit", "-m", "Update models and database"], cwd=os.getcwd(), capture_output=True, text=True)
        if result.returncode == 0 or "nothing to commit" in result.stdout:
            subprocess.run(["git", "push", "origin", "main"], cwd=os.getcwd(), check=True, capture_output=True, text=True)
            logging.info("Successfully pushed changes to GitHub.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Git commit/push failed: {e.stderr}")
