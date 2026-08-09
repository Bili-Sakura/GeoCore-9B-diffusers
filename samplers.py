"""Euler flow-matching samplers used during training-time validation."""

import torch


def euler_sampler_flux2(
        model,
        latents,
        cond_kwargs,
        uncond_kwargs,
        num_steps=20,
        heun=False,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
):
    _dtype = latents.dtype
    t_steps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float64)
    x_next = latents.to(torch.float64)
    device = x_next.device

    def get_model_input(x, t, c_kwargs, u_kwargs):
        if cfg_scale > 1.0 and guidance_low <= t <= guidance_high:
            model_input = torch.cat([x] * 2, dim=0)
            t_input = torch.cat([t.expand(x.shape[0])] * 2, dim=0)
            combined_kwargs = {}
            for k in c_kwargs.keys():
                if k in ["ctx", "y"]:
                    combined_kwargs[k] = torch.cat([c_kwargs[k], u_kwargs[k]], dim=0)
                elif k in ["res", "lon", "lat"]:
                    val_u = u_kwargs.get(k, c_kwargs[k])
                    combined_kwargs[k] = torch.cat([c_kwargs[k], val_u], dim=0)
                elif k in ["x_ids", "ctx_ids"]:
                    if c_kwargs[k].dim() == 3 and c_kwargs[k].shape[0] == x.shape[0]:
                        combined_kwargs[k] = torch.cat([c_kwargs[k]] * 2, dim=0)
                    else:
                        combined_kwargs[k] = c_kwargs[k]
                else:
                    combined_kwargs[k] = c_kwargs[k]
            out = model(model_input.to(_dtype), combined_kwargs.pop("x_ids"), t_input.to(_dtype), **combined_kwargs)
            if isinstance(out, tuple):
                out = out[0]
            out_cond, out_uncond = out.chunk(2)
            return out_uncond.to(torch.float64) + cfg_scale * (out_cond.to(torch.float64) - out_uncond.to(torch.float64))

        t_input = t.expand(x.shape[0])
        k_copy = {**c_kwargs}
        x_ids = k_copy.pop("x_ids")
        out = model(x.to(_dtype), x_ids, t_input.to(_dtype), **k_copy)
        if isinstance(out, tuple):
            out = out[0]
        return out.to(torch.float64)

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            d_cur = get_model_input(x_cur, t_cur.to(device), cond_kwargs, uncond_kwargs)
            x_next = x_cur + (t_next - t_cur) * d_cur
            if heun and (i < num_steps - 1):
                d_prime = get_model_input(x_next, t_next.to(device), cond_kwargs, uncond_kwargs)
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next.to(_dtype)
