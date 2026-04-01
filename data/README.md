# Data Directory

This directory stores local data assets used during development.

Contents are not versioned by default because they may be large, regenerated, licensed separately, or machine-specific.

Typical contents include:

- simulation files
- intermediate local datasets
- temporary benchmark inputs prepared outside the repository

If a dataset becomes essential for public reproducibility, it should eventually be:

- documented in the main README
- referenced with provenance and acquisition instructions
- packaged separately or replaced by a lightweight reproducible example
