# Data Processing and Plotting (Exercise 3.2)

Professional Python solution for Part II Chemistry Programming Practical Exercise 3.2:
- parse Gaussian output files for a triatomic molecule,
- extract geometry and SCF energies,
- find equilibrium geometry,
- plot the potential energy surface,
- estimate symmetric stretch and bend vibrational frequencies in `cm^-1`.

## Project structure

- `src/data_processing_plotting/surface.py` core parsing + analysis logic
- `src/data_processing_plotting/cli.py` command-line interface
- `tests/` unit tests with synthetic Gaussian-like files

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

### Local directory input

```bash
ex2-surface /path/to/H2Ooutfiles --plot h2o_surface.png
```

### URL input (best effort)

```bash
ex2-surface "https://example.com/some-directory/"
```

If a URL is used, the tool tries to discover `.out` links from the page and download them.

## Output

The CLI prints:
- equilibrium energy (Hartree),
- equilibrium bond length (Angstrom),
- equilibrium bond angle (degrees),
- estimated stretch and bend frequencies (`cm^-1`).

It also writes a 3D surface plot image.

## Tests

```bash
pytest -q
```

## GitHub push

```bash
git init
git add .
git commit -m "Exercise 3.2: Gaussian data processing and PES plotting"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

