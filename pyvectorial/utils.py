
import logging as log
import sbpy.activity as sba
import astropy.units as u


def print_radial_density(vmodel):
    print("\n\nRadius (km) vs Fragment density (1/cm3)\n---------------------------------------")
    rgrid = vmodel['radial_grid']
    dens = vmodel['radial_density']
    for r, n_r in zip(rgrid, dens):
        print(f"{r.to(u.km):10.1f} : {n_r.to(1/(u.cm**3)):8.4f}")


def print_column_density(vmodel):
    print("\nRadius (km) vs Column density (1/cm2)\n-------------------------------------")
    cds = list(zip(vmodel['column_density_grid'], vmodel['column_densities']))
    for pair in cds:
        print(f'{pair[0].to(u.km):7.0f} :\t{pair[1].to(1/(u.cm*u.cm)):5.3e}')


def print_binned_times(production_dict):
    print("")
    print("Binned time production summary")
    print("------------------------------")
    for q, t in zip(production_dict['q_t'], production_dict['times_at_productions']):
        t_u = t.to(u.day).value
        print(f"Q: {q}\t\tt_start (days ago): {t_u}")
    print("")


def show_fragment_agreement(vmodel):
    print("\nFragment agreement check:")
    print(f"\tTheoretical total number of fragments in coma:\t\t\t {vmodel['num_fragments_theory']:.7e}")
    print(f"\tTotal number of fragments from density grid integration:\t {vmodel['num_fragments_grid']:.7e}")


def show_aperture_checks(coma):

    log.debug("Starting aperture checks ...")
    f_theory = coma.vmodel['num_fragments_theory']

    # use various large apertures to see how much we recover
    ap1 = sba.RectangularAperture((coma.vmodel['max_grid_radius'].value, coma.vmodel['max_grid_radius'].value) * u.m)
    ap2 = sba.CircularAperture((coma.vmodel['max_grid_radius'].value) * u.m)
    ap3 = sba.AnnularAperture([500000, coma.vmodel['max_grid_radius'].value] * u.m)

    rect_result = coma.total_number(ap1)*100/f_theory
    circular_result = coma.total_number(ap2)*100/f_theory
    annular_result = coma.total_number(ap3)*100/f_theory

    print("\nPercent of fragments recovered by integrating column density over")
    print(f"\tLarge rectangular aperture:\t\t\t\t\t\t{rect_result:2.2f}%")
    print(f"\tLarge circular aperture:\t\t\t\t\t\t{circular_result:2.2f}%")
    print(f"\tAnnular aperture, inner radius 500000 km, outer radius of entire grid:\t{annular_result:2.2f}%")
