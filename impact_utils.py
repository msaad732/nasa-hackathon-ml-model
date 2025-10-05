# impact_utils.py (Physics-only version)
import math

DEFAULT_DENSITY = 3000.0  # kg/m^3 (rocky asteroid)

def diameter_to_mass(diameter_m, density_kg_m3=DEFAULT_DENSITY):
    """Convert asteroid diameter (m) to mass (kg)."""
    r = diameter_m / 2.0
    vol = (4.0/3.0) * math.pi * (r**3)
    return vol * density_kg_m3

def kinetic_energy(mass_kg, velocity_m_s):
    """Kinetic energy (Joules)."""
    return 0.5 * mass_kg * velocity_m_s**2

def energy_to_magnitude(E_joules):
    """Convert energy (J) to earthquake magnitude (Richter)."""
    return (math.log10(E_joules) - 4.8) / 1.5

def impact_from_diameter(diameter_m, velocity_m_s, density=DEFAULT_DENSITY):
    """Compute impact effects directly from diameter + velocity."""
    mass = diameter_to_mass(diameter_m, density)
    E_kin = kinetic_energy(mass, velocity_m_s)
    M = energy_to_magnitude(E_kin)
    return {
        "mass_kg": mass,
        "kinetic_E_J": E_kin,
        "magnitude": M,
    }

def impact_from_mass(mass_kg, velocity_m_s):
    """Compute impact effects directly from mass + velocity."""
    E_kin = kinetic_energy(mass_kg, velocity_m_s)
    M = energy_to_magnitude(E_kin)
    return {
        "mass_kg": mass_kg,
        "kinetic_E_J": E_kin,
        "magnitude": M,
    }
