import galax.dynamics as gd
import quaxed.numpy as jnp
import jax
import unxt as u

__all__ = ["integrate_pop_with_events"]

@jax.jit
def _integrate_star_with_events(field, x0, v0, t0, events, tf):

    ev_t_no_units = u.ustrip(field.units, events["time"])
    ev_dv_no_units = u.ustrip(field.units, events["delta_v"])

    x, v = x0.ustrip(field.units), v0.ustrip(field.units)
    # print(x, v)

    sol = gd.integrate_field(field, (x, v), jnp.stack((t0, events["time"][0])), max_steps=100000)
    # print("First: ", sol.ys[0][-1], sol.ys[1][-1])
    # print(sol.ts)

    x, v = sol.ys[0][-1], sol.ys[1][-1]
    v = v + ev_dv_no_units[0]
    sol = gd.integrate_field(
        field, (x, v), jnp.stack((sol.ts[-1], ev_t_no_units[1])), max_steps=100000
    )
    # print(sol.ts)

    # print("Second: ", sol.ys[0][-1], sol.ys[1][-1])

    x, v = sol.ys[0][-1], sol.ys[1][-1]
    v = v + ev_dv_no_units[1]
    sol = gd.integrate_field(
        field, (x, v), jnp.stack((sol.ts[-1], tf.uconvert(field.units).value)), max_steps=100000
    )
    # print(sol.ts)

    # print("Final: ", sol.ys[0][-1], sol.ys[1][-1])
    return (sol.ys[0][-1], sol.ys[1][-1])


_integrate_binary_with_events = jax.vmap(
    _integrate_star_with_events, in_axes=(None, 0, 0, 0, 0, None)
)

integrate_pop_with_events = jax.vmap(
    _integrate_binary_with_events, in_axes=(None, 0, 0, 0, 0, None)
)