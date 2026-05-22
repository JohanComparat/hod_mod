# Contributing

## Development setup

```bash
git clone https://github.com/JohanComparat/hod_mod.git
cd hod_mod
pip install -e ".[dev]"
```

## Running the test suite

```bash
JAX_PLATFORMS=cpu python -m pytest tests/ -v
```

Tests must not require internet access or trained model weights.

## Adding a new HOD model

1. Subclass `HODBase` in `hod_mod/galaxies/hod.py`:

   ```python
   class MyHODModel(HODBase):
       def nc_ns(self, log10m_arr, hod_params):
           # return (N_cen array, N_sat array) on log10m_arr
           ...

       @staticmethod
       def default_params():
           return {"log10mmin": 12.0, ...}
   ```

2. Export from `hod_mod/galaxies/__init__.py`.

3. Add to the `HOD_MODELS` dict in `hod_mod/fitting/hod_wp.py`.

4. Add tests in `tests/test_galaxies.py` (occupation range checks,
   `_integrate()` returning finite positive values).

5. If the model takes only `hmf` in its constructor (no separate `halo_bias`
   argument), set `_SINGLE_ARG_INIT = True` as a class attribute.

## Code style

- Python 3.11+; all numerical code must be JAX-compatible (`jnp.*`, `@jax.jit`).
- No comments unless the *why* is non-obvious.
- Docstrings must state the physical quantity, units, and the equation
  implemented (LaTeX math rendered by Sphinx + MathJax).
- No `print` statements in library code; only in `scripts/`.
- All physical units are h-units (Mpc/h, M_sun/h) unless explicitly noted.
