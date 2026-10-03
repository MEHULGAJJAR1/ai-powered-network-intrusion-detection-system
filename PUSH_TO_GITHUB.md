# GitHub repository

This project has been published to the public GitHub repository:

**<https://github.com/MEHULGAJJAR1/ai-powered-network-intrusion-detection-system>**

Clone it with:

```bash
git clone https://github.com/MEHULGAJJAR1/ai-powered-network-intrusion-detection-system.git
cd ai-powered-network-intrusion-detection-system
```

## Push later changes

To push additional commits, authenticate with GitHub CLI and push to `main`:

```bash
gh auth login --hostname github.com --git-protocol https --web
git add .
git commit -m "Describe your change"
git push origin main
```

GitHub will authenticate through the browser/device flow. Do not put personal access tokens in source files, chat, or remote URLs.

## What is excluded

The `.gitignore` excludes `.env`, downloaded dataset files, generated model/joblib artifacts, runtime history databases, Python caches and synthetic demo CSVs. The repository contains `.env.example` and instructions to download KDD Cup 1999 and train locally. Review licensing, data sensitivity, model provenance and GitHub rules before adding datasets or artifacts.
