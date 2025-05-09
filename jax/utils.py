import jax
import jax.numpy as jnp

__all__ = ()


def laplacian(f):
    def lap(x):
        grad_f = jax.grad(f)
        df, grad_f_jvp = jax.linearize(grad_f, x)
        eye = jnp.eye(len(x))
        d2f = jnp.diag(jax.vmap(grad_f_jvp)(eye))
        return jnp.sum(d2f), df

    return lap


def masked_mean(x, mask):
    x = jnp.where(mask, x, 0)
    return x.sum() / jnp.sum(mask)


def pairwise_distance(coords1, coords2):
    return jnp.linalg.norm(coords1[..., :, None, :] - coords2[..., None, :, :], axis=-1)


def pairwise_diffs(coords1, coords2):
    diffs = coords1[..., :, None, :] - coords2[..., None, :, :]
    return jnp.concatenate([diffs, (diffs**2).sum(axis=-1, keepdims=True)], axis=-1)


def pairwise_self_distance(coords, full=False):
    i, j = jnp.triu_indices(coords.shape[-2], k=1)
    diffs = coords[..., :, None, :] - coords[..., None, :, :]
    dists = jnp.linalg.norm(diffs[..., i, j, :], axis=-1)
    if full:
        dists_full = jnp.zeros(diffs.shape[:-1])
        dists_full = dists_full.at[..., i, j].set(dists)
        dists_full = dists_full.at[..., j, i].set(dists)
        dists = dists_full
    return dists


def triu_flat(x):
    i, j = jnp.triu_indices(x.shape[1], x.shape[1], 1)
    return x[..., i, j]


def nuclear_energy(mol):
    coords, charges = mol.coords, mol.charges
    coulombs = (
        charges[:, None] * charges / jnp.linalg.norm(coords[:, None] - coords, axis=-1)
    )
    return jnp.triu(coulombs, 1).sum()


def nuclear_potential(rs, mol):
    dists = jnp.linalg.norm(rs[..., :, None, :] - mol.coords, axis=-1)
    return -(mol.charges / dists).sum(axis=(-1, -2))


def electronic_potential(rs):
    i, j = jnp.triu_indices(rs.shape[-2], k=1)
    dists = jnp.linalg.norm(
        (rs[..., :, None, :] - rs[..., None, :, :])[..., i, j, :], axis=-1
    )
    return (1 / dists).sum(axis=-1)
