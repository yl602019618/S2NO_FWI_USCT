"""Original convergent Born series solver."""

import torch
import sympy
from ._config import Bunch
import math
import torch.nn.functional as F
import numpy as np
import time


### simluateSetting and relativce function
class SimulateSetting:
    def __init__(self, speed, CBSConfig) -> None:
        # basic CBSconfig
        pixel_size = CBSConfig.lamb / CBSConfig.PPW
        boundary_widths = CBSConfig.boundary_widths * CBSConfig.PPW
        scale = CBSConfig.dh / pixel_size
        speed_mean = torch.mean(speed)  #
        adjusted_all_size, roi, bw_size, Flag_dim = ensure_3d_shape(
            speed, boundary_widths
        )
        speed = expand_dims(speed, Flag_dim)

        lamb = speed_mean / (CBSConfig.freq * scale)

        # Medium
        refractive_index = speed_mean / speed
        e_r = refractive_index**2
        e_r_min = torch.min(e_r)
        e_r_max = torch.max(e_r)
        e_r_center = (e_r_max + e_r_min) / 2

        k0 = 2 * torch.pi / lamb  # wavenumber in empty medium
        k0c = (
            torch.sqrt(e_r_center) * k0
        )  # wavenumber for center value refractive index

        ## padding step
        # adjusted_all_size, roi, bw_size, Flag_dim = ensure_3d_shape(
        #     e_r, boundary_widths
        # )
        self.Grid = Get_grid(pixel_size, adjusted_all_size)
        V, epsilon = RI2potential(e_r, k0, k0c, adjusted_all_size, roi, bw_size)
        gamma = (
            1j / epsilon
        ) * V  # calculate potential map with background gain (gamma)
        mix_weight1 = 1 - gamma
        mix_weight2 = 1j * gamma**2
        mix_weight1 = mix_weight1.unsqueeze(0).repeat(1, 1, 1, 1)
        mix_weight2 = mix_weight2.unsqueeze(0).repeat(1, 1, 1, 1)
        iterations_per_cycle = lamb / (
            2 * k0c / epsilon
        )  # divide wavelength by pseudo-propagation length
        k02e = k0c**2 / epsilon + 1j
        wiggles = compute_wiggles(self.Grid, Flag_dim, epsilon, k02e)
        N_wiggle = len(wiggles)

        # prepare state (contains all data that is unique to a single run of the simulation)
        self.state = Bunch(
            {
                "energy_calculation_interval": 16,
                "iteration": 0,
                "max_iterations": math.ceil(CBSConfig.Bornmax * iterations_per_cycle*3),
                "diff_energy": {},
                "last_step_energy": torch.inf,
                "calculate_energy": True,
                "has_next": True,
            }
        )

        CBSConfig.update(
            {
                "Flag_dim": Flag_dim,
                "wiggles": wiggles,
                "N_wiggle": N_wiggle,
                "gamma": gamma.cuda(),
                "mix_weight1": mix_weight1.cuda(),
                "mix_weight2": mix_weight2.cuda(),
                "k02e": k02e,
                "epsilon": epsilon,
                "roi": roi,
            }
        )
        self.CBSConfig = CBSConfig


def expand_dims(tensor, flag_dim):
    for dim, flag in enumerate(flag_dim):
        if not flag:
            tensor = tensor.unsqueeze(dim)
    return tensor


def Get_grid(pixel_size, adjusted_all_size):
    x = pixel_size * torch.arange(adjusted_all_size[0]).reshape(-1, 1, 1).cuda()
    y = pixel_size * torch.arange(adjusted_all_size[1]).reshape(1, -1, 1).cuda()
    z = pixel_size * torch.arange(adjusted_all_size[2]).reshape(1, 1, -1).cuda()

    # df * dt = 1 / N
    dfx = 2 * torch.pi / (pixel_size * adjusted_all_size[0])  # why  2 * pi
    dfy = 2 * torch.pi / (pixel_size * adjusted_all_size[1])  # why  2 * pi
    dfz = 2 * torch.pi / (pixel_size * adjusted_all_size[2])  # why  2 * pi

    fx = (
        dfx
        * torch.fft.fftshift(
            torch.arange(adjusted_all_size[0])
            - torch.floor(torch.tensor(adjusted_all_size[0] / 2))
        )
        .reshape(-1, 1, 1)
        .cuda()
    )

    fy = (
        dfy
        * torch.fft.fftshift(
            torch.arange(adjusted_all_size[1])
            - torch.floor(torch.tensor(adjusted_all_size[1] / 2))
        )
        .reshape(1, -1, 1)
        .cuda()
    )

    fz = (
        dfz
        * torch.fft.fftshift(
            torch.arange(adjusted_all_size[2])
            - torch.floor(torch.tensor(adjusted_all_size[2] / 2))
        )
        .reshape(1, 1, -1)
        .cuda()
    )

    Grid = Bunch(
        {
            "adjusted_all_size": adjusted_all_size,
            "pixel_size": pixel_size,
            "dfx": dfx,
            "x": x,
            "fx": fx,
            "dfy": dfy,
            "y": y,
            "fy": fy,
            "dfz": dfz,
            "z": z,
            "fz": fz,
        }
    )
    return Grid


def apply_edge_filters(V, N, roi, boundary_widths):
    Bl, Br = roi
    for dim in range(len(N)):
        bl = Bl[dim]  # width of added boundary
        br = Br[dim]
        roi_size = N[dim] - bl - br
        if bl > 0:
            L = boundary_widths[dim]  # width of boundary layer
            if all(n > 1 for n in N):  # use linear window designed for 3D simulations
                # define fun
                window = lambda B: ((torch.arange(1, B + 1) - 0.21) / (B + 0.66))
            else:  # use nuttall window for 1D and 2D simulations
                # define fun
                win = lambda nutall, B: nutall[:B]
                window = lambda B: win(nuttallwin(2 * L - 1), B)

            # Add a zero to the window if number of grid points is even
            if br == bl:
                smoothstep = torch.cat([torch.tensor([0.0]), window(L - 1)])
            else:  # For odd number of grid points, zeros are already added
                smoothstep = window(L)
            # Construct the filter
            filt = torch.cat(
                [
                    torch.zeros(bl - L),
                    smoothstep,
                    torch.ones(roi_size),
                    torch.flipud(smoothstep),
                    torch.zeros(br - L),
                ]
            ).cuda()

            # Reshape filter to match the required dimension
            filter_shape = [1] * V.dim()
            filter_shape[dim] = filt.size(0)
            filt = filt.view(filter_shape)

            # Apply filter to potential map
            V = V * filt
    return V


def shift_src(src, CBSConfig):
    roi = CBSConfig.roi
    epsilon = CBSConfig.epsilon
    N_wiggle = len(CBSConfig.wiggles)

    Bl, _ = roi
    A = 1j / epsilon / N_wiggle
    shifted_src = []
    for item in src:
        position = [item["ix"] + Bl[0], item["iy"] + Bl[1], item["iz"] + Bl[2]]
        value = item["value"] * A
        energy = abs(value) ** 2
        shifted_src.append(
            Bunch({"position": position, "value": value, "energy": energy})
        )
    return shifted_src, N_wiggle


def validate_source_positions(source, roi, adjusted_all_size):
    Bl, Br = roi
    for item in source:
        for index_dim in range(len(item.position)):
            assert (
                item.position[index_dim] >= Bl[index_dim]
                and item.position[index_dim]
                <= adjusted_all_size[index_dim] - Br[index_dim]
            ), f"source.index={item.position[index_dim]} is out of range [{Bl[index_dim]}, {adjusted_all_size[index_dim] - Br[index_dim]}]"


def compute_wiggles(Grid, Flag_dim, epsilon, k02e):
    wiggle_set = wiggle_perm(Flag_dim)
    # Calculate phase ramps and coordinates for every different wiggle
    Nwiggles = wiggle_set.size(1)
    wiggle_descriptors = [None] * Nwiggles  # Pre-allocate memory
    for w_i in range(Nwiggles):
        wiggle_descriptors[w_i] = wiggle_descriptor(
            Grid, wiggle_set[:, w_i], epsilon, k02e
        )
    return wiggle_descriptors


def wiggle_perm(wiggle_flags):
    """
    Returns matrix with all possible combinations of enabled wiggle directions.
    """
    n_directions = len(wiggle_flags)  # Number of wiggle directions considered
    n_wiggles = sum(wiggle_flags)  # Number of enabled wiggle flags
    wiggle_set = torch.zeros((n_directions, 2**n_wiggles))
    # Powers of minus one generate rows of alternating ones and minus ones.
    wiggle_set[wiggle_flags, :] = (-1) ** (
        torch.ceil(
            torch.arange(1, 2**n_wiggles + 1) / 2 ** torch.arange(n_wiggles).view(-1, 1)
        )
        + 1
    )
    return wiggle_set


def wiggle_descriptor(grid, wig, epsilon, k02e):
    """
    Calculate phase ramps and coordinates for a given wiggle permutation.
    determine all permutations of wiggle directions
    """
    wd = Bunch({})
    sqrt_epsilon = math.sqrt(epsilon)
    # Construct coordinates, shift quarter of a pixel when wiggling
    ### using python idnex style
    wd.pxe = (grid.fx - grid.dfx * wig[0] / 4) / sqrt_epsilon
    wd.pye = (grid.fy - grid.dfy * wig[1] / 4) / sqrt_epsilon
    wd.pze = (grid.fz - grid.dfz * wig[2] / 4) / sqrt_epsilon

    # Construct real space phase gradients to compensate for the pixel shift in k_space
    wd.gx = torch.exp(
        (1j * wig[0] * grid.x) * (torch.pi / 2) / (grid.pixel_size * grid.x.shape[0])
    ).cuda()
    wd.gy = torch.exp(
        (1.0j * wig[1] * grid.y) * (torch.pi / 2) / (grid.pixel_size * grid.y.shape[1])
    ).cuda()
    wd.gz = torch.exp(
        (1.0j * wig[2] * grid.z) * (torch.pi / 2) / (grid.pixel_size * grid.z.shape[2])
    ).cuda()

    wd.filter_step0 = wd.gx * wd.gy * wd.gz
    wd.scaler_step1 = wd.pxe**2 + wd.pye**2 + wd.pze**2 - k02e
    wd.fiter_step2 = wd.gx.conj() * wd.gy.conj() * wd.gz.conj()
    wd.filter_step0.unsqueeze(0).repeat(1, 1, 1, 1)
    wd.scaler_step1.unsqueeze(0).repeat(1, 1, 1, 1)
    wd.fiter_step2.unsqueeze(0).repeat(1, 1, 1, 1)

    return wd


def RI2potential(e_r, k0, k0c, adjusted_all_size, roi, bw_size, epsilonmin=3):

    padded_e_r = add_boundary_layer(e_r, roi)
    #    determine largest absolute potential fluctuation of all submedia to guarantee convergence
    V_tot = padded_e_r * k0**2 - k0c**2

    Vabs_max = torch.max(torch.abs(V_tot))

    # Get the index of the maximum absolute value in V_tot
    max_index = torch.argmax(torch.abs(V_tot))

    # If Vabs_max corresponds to a point with only absorption
    # and we use epsilon = Vabs_max, then at that point V will be 0,
    # and the method does not work. In this special case, add a small offset to epsilon
    if torch.real(V_tot.flatten()[max_index]) < 0.05 * Vabs_max:
        Vabs_max *= 1.05

    # Ensure epsilon is at least epsilonmin to avoid divergence when simulating an empty medium
    epsilon = max(Vabs_max.item(), epsilonmin)

    V = padded_e_r * k0**2 - k0c**2 - 1j * epsilon  # calculate potential map

    # import matplotlib.pyplot as plt
    # plt.imshow( torch.abs(V).cpu(),origin='lower' )
    # plt.colorbar()
    # plt.savefig('./V.png')
    # plt.close()

    V_filtered = apply_edge_filters(V, adjusted_all_size, roi, bw_size)

    # import matplotlib.pyplot as plt
    # plt.imshow( torch.abs(V_filtered).cpu(),origin='lower' )
    # plt.colorbar()
    # plt.savefig('./V_filtered.png')
    # plt.close()

    return V_filtered, epsilon


def add_boundary_layer(e_r, roi):
    Bl, Bu = roi
    p3d = tuple([item for pair in zip(Bl[::-1], Bu[::-1]) for item in pair])
    result_e_r = F.pad(e_r.unsqueeze(0), p3d, mode="replicate")[0]
    return result_e_r


def ensure_3d_shape(tensor, boundary_widths: int):
    """
    Ensure the input tensor has a 3D shape by adjusting dimensions and calculating boundary sizes.

    Args:
        tensor: Input tensor.
        boundary_widths: Widths to apply on the boundaries.

    Returns:
        adjusted_all_size: The adjusted size of the tensor.
        roi: Regions of interest (left and right boundaries).
        bw_size: The boundary sizes.
        flag_dim: Flags indicating dimensions of the original tensor.
    """

    shape = tensor.shape
    original_dim = len(shape)

    if original_dim > 3:
        raise ValueError(
            "The tensor has more than 3 dimensions, which is not supported."
        )

    # Determine 3D shape and boundary sizes
    three_shape = shape + (1,) * (3 - original_dim)
    boundary_sizes = [
        boundary_widths if d else 0
        for d in [original_dim >= 1, original_dim >= 2, original_dim == 3]
    ]

    # Calculate adjusted sizes
    adjusted_all_size = adjust_sizes(
        [dim + 2 * bw for dim, bw in zip(three_shape, boundary_sizes)],
        [original_dim >= d for d in range(1, 4)],
    )

    # Calculate left and right boundaries
    low_boundaries, up_boundaries = zip(
        *(
            (
                math.ceil((adjusted - original) / 2),
                math.floor((adjusted - original) / 2),
            )
            for adjusted, original in zip(adjusted_all_size, three_shape)
        )
    )
    roi = [list(low_boundaries), list(up_boundaries)]

    return (
        adjusted_all_size,
        roi,
        boundary_sizes,
        [original_dim >= d for d in range(1, 4)],
    )


def adjust_sizes(all_size, flag_dim):
    """
    Adjust sizes based on the specified flags and ensure factors are within certain limits.

    Args:
        all_size: List of sizes to adjust.
        flag_dim: List of flags indicating whether to adjust the corresponding size.

    Returns:
        List of adjusted sizes.
    """

    assert len(all_size) == len(
        flag_dim
    ), "The lengths of all_size and flag_dim must match."

    for i, item in enumerate(all_size):
        if not flag_dim[i]:
            continue

        while True:
            factors = list(sympy.factorint(item).keys())
            if max(factors) <= 11 and (len(factors) <= 2 or sorted(factors)[-2] <= 5):
                break
            item += 1

        all_size[i] = item

    return all_size


def nuttallwin(N, sflag="symmetric"):
    """
    NUTTALLWIN Nuttall defined minimum 4-term Blackman-Harris window.

    Args:
        N (int): Number of points in the window.
        sflag (str): Window sampling, either 'symmetric' or 'periodic'.
                     Default is 'symmetric'.

    Returns:
        torch.Tensor: The N-point modified minimum 4-term Blackman-Harris window.
    """
    # Validate input
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")

    if sflag not in ["symmetric", "periodic"]:
        raise ValueError("sflag must be either 'symmetric' or 'periodic'.")

    # Coefficients from Nuttall's paper
    a = [0.3635819, 0.4891775, 0.1365995, 0.0106411]

    if sflag == "periodic":
        # x = torch.arange(0, N).cuda() * 2.0 * torch.pi / N
        x = torch.arange(0, N) * 2.0 * torch.pi / N
    else:  # symmetric
        # x = torch.arange(0, N).cuda() * 2.0 * torch.pi / (N - 1)
        x = torch.arange(0, N) * 2.0 * torch.pi / (N - 1)

    # Compute the window
    w = a[0] - a[1] * torch.cos(x) + a[2] * torch.cos(2 * x) - a[3] * torch.cos(3 * x)

    return w


### CBS solve and relativce function


class CBS:
    def __init__(self, speed, CBSconfig) -> None:
        simulatsetting = SimulateSetting(speed, CBSconfig)
        self.CBSConfig = simulatsetting.CBSConfig
        self.state = simulatsetting.state
        self.Grid = simulatsetting.Grid
        self.get_croped_field = Get_croped_field(
            self.CBSConfig.Flag_dim, self.CBSConfig.roi
        )

    def solve(self, src, Batch=1):
        Return_E = []
        energy_threshold = self.CBSConfig.energy_threshold
        self.Grid.adjusted_all_size.insert(0, 1)

        Real_Batch = min(len(src), Batch)
        self.Grid.adjusted_all_size[0] = Real_Batch

        src_mini_batch, N_wiggle = shift_src(src, self.CBSConfig)

        dh = self.CBSConfig.dh
        # simulation iterations
        dE = torch.from_numpy(
            np.zeros(self.Grid.adjusted_all_size, dtype=np.complex64)
        ).cuda()
        E = dE.clone()

        get_dE_V2 = Get_dE(self.CBSConfig.mix_weight1, self.CBSConfig.mix_weight2)
        has_next = True
        iteration = 0
        while has_next:
            # add source term (first Nwiggles iterations only)
            if (
                # self.state.iteration < N_wiggle
                iteration
                < N_wiggle
            ):  # During the first few iterations: add source term
                Etmp = add_to(src_mini_batch, dE)
            else:
                Etmp = dE.clone()

            # select correct wiggle and medium number based on iteration number
            index_wiggle = iteration % N_wiggle
            wig = self.CBSConfig.wiggles[index_wiggle]
            # dE = get_dE(
            #     dE, Etmp, wig, self.CBSConfig.mix_weight1, self.CBSConfig.mix_weight2
            # )
            # t_start1 = time.time()
            dE = get_dE_V2(dE, Etmp, wig)
            # print(f"get_dE_V2 time: {time.time() - t_start1}")
            E += dE

            croped_field = self.get_croped_field(dE)
            # import matplotlib.pyplot as plt
            # # plt.plot(torch.sum(torch.abs(croped_field) ** 2 , dim=(1,2,3)).cpu())
            # # plt.savefig('./test.png')
            # # plt.close()
            # plt.imshow(torch.real(croped_field[0,...,0]).cpu(),origin='lower')
            # plt.colorbar()
            # plt.savefig('./test.png')
            # plt.close()

            last_step_energy = torch.min(
                torch.sum(torch.abs(croped_field) ** 2, dim=(1, 2, 3))
            )

            can_terminate = iteration % N_wiggle
            if iteration == 0:
                init_energy = last_step_energy
            if can_terminate == 0 and (
                last_step_energy / init_energy < energy_threshold  # converged
                or iteration >= self.CBSConfig.Bornmax  # unconverged
            ):
                has_next = False

            iteration += 1

        # Final computional steps divide field by gamma to convert E' -> E
        E = E / self.CBSConfig.gamma
        scaled_E = -torch.conj(self.get_croped_field(E)) / dh / dh * 2.5e10

        return torch.squeeze(scaled_E).permute(1, 2, 0)


def add_to(source, E):

    E_source = E.clone()

    for index_src in range(len(source)):
        item = source[index_src]
        x, y, z = item.position
        # E_source[x, y, z] = E_source[x, y, z] + source.value
        E_source[index_src, x, y, z] += item.value
    return E_source


def Get_dE(mix_weight1, mix_weight2):
    def fun(dE, Etmp, wig):
        # step0 - transform to Fourier domain
        E_filtered = Etmp * wig.filter_step0
        # applied fft
        fftE = torch.fft.fftn(E_filtered, dim=(1, 2, 3))

        # step1 - apply propagation kernel
        fftE = fftE / wig.scaler_step1

        # step2 - Perform modified inverse Fourier transform (reverses wiggle phase ramps)
        fftE = torch.fft.ifftn(fftE, dim=(1, 2, 3))
        fftE = fftE * wig.fiter_step2

        # step3 - Mix
        Ediff = mix_weight1 * dE - mix_weight2 * fftE

        return Ediff

    return fun


# def Get_croped_field(Flag_dim, roi):
#     Bl, Br = roi
#     if Flag_dim == [True, False, False]:

#         def fun(dE):
#             return torch.squeeze(dE[Bl[0] : -Br[0], :, :])

#         return fun
#     elif Flag_dim == [True, True, False]:

#         def fun(dE):
#             return torch.squeeze(dE[Bl[0] : -Br[0], Bl[1] : -Br[1], :])

#         return fun
#     elif Flag_dim == [True, True, True]:

#         def fun(dE):
#             return torch.squeeze(dE[Bl[0] : -Br[0], Bl[1] : -Br[1], Bl[2] : -Br[2]])

#         return fun


def Get_croped_field(Flag_dim, roi):
    Bl, Bu = roi

    def fun(dE):
        cropped_dE = dE[
            :,
            Bl[0] : -Bu[0] if Bu[0] > 0 else None,
            Bl[1] : -Bu[1] if Bu[1] > 0 else None,
            Bl[2] : -Bu[2] if Bu[2] > 0 else None,
        ]
        return cropped_dE

    return fun
