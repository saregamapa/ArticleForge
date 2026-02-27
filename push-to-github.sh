#!/bin/bash
# Run this script in Terminal to push ArticleForge to GitHub.
# Uses your SSH key (git@github.com) or will prompt for HTTPS credentials.
set -e
cd "$(dirname "$0")"
git remote set-url origin git@github.com:saregamapa/ArticleForge.git
git push -u origin main
echo "Done! View at https://github.com/saregamapa/ArticleForge"
