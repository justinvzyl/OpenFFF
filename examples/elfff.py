import numpy as np
from collections import namedtuple

num_particles = 10_000
boltzmann_constant = 1.380649E-23 # J/K
dynamic_viscosity_carrier = 0.001  # Pa*s
temperature = 293.15  # K
particle_diameter = 10E-9  # 10 nm
electrophoretic_mobility = -4.0E-8  # µm X cm / (V X s) E-6E-2 = E-8
dt = 0.5  # seconds
carrier_velocity = 1.0E-2 / 60.0  # m/s (1 ml/ min = 1cm^3/60 seconds, 1 cm /60 seconds)
E = -25 # V/m

diffusion_coeff = (temperature * boltzmann_constant) / (3 * np.pi * dynamic_viscosity_carrier * particle_diameter)

channel_width = 3.2E-2  # cm
channel_height = 178E-6  # µm

theoretical = diffusion_coeff / (electrophoretic_mobility * E)
print(f'Theoretical height: {theoretical:.2E}')

ColumnIO = namedtuple("ColumnIO", "x1 x2 y1 y2")

inlet = ColumnIO(0, 0.75E-3, channel_height, channel_height)

outlet_x = channel_width
outlet_y = (0, channel_height)

# randomly distribute particles at the inlet
Particles = namedtuple("Particles", "x y")
particles = Particles(x=np.random.rand(num_particles) * inlet.x2,
                      y=np.random.rand(num_particles) * inlet.y2)

# calculate the initial average height
y_avg = sum(particles.y) / num_particles
x_avg = sum(particles.x) / num_particles
print(f'Average particle cloud height t=0 is\t{y_avg:.2E} m')
print(f'Average particle cloud width t=0 is\t\t{x_avg:.2E} m')

def check_boundary_conditions(particles: Particles) -> Particles:
    # top
    y = particles.y
    particles_past_boundary = np.where(particles.y > channel_height)
    new_y = channel_height - (particles.y[particles_past_boundary] - channel_height)
    y[particles_past_boundary] = new_y
    # bottom
    particles_past_boundary = np.where(particles.y < 0)
    new_y = -1.0 * particles.y[particles_past_boundary]
    y[particles_past_boundary] = new_y
    #left
    x = particles.x
    particles_past_boundary = np.where(particles.x < 0)
    new_x = -1.0 * particles.x[particles_past_boundary]
    x[particles_past_boundary] = new_x
    return Particles(x, y)


def step(particle_cloud: Particles) -> Particles:
    # propagate x coordinates
    # x(t+dt) = x(t) + n * ld(dt) + vpx * dt
    n = np.random.normal(loc=0, scale=1, size=num_particles)
    ld = np.sqrt(2.0 * diffusion_coeff * dt)
    vpx = 6.0 * carrier_velocity * (
                np.divide(particle_cloud.y, channel_height) - np.square(np.divide(particle_cloud.y, channel_height)))
    x = particle_cloud.x + n * ld + vpx * dt
    # propagate y coordinates
    # y(t+dt) = y(t) + n * ld(dt) + vpy * dt
    vpy = electrophoretic_mobility * E
    y = particle_cloud.y + n * ld + vpy * dt

    return check_boundary_conditions(Particles(x, y))

t = 0
while t <= 300:
    particles = step(particles)
    t += dt

y_avg = sum(particles.y)/num_particles
x_avg = sum(particles.x)/num_particles
print(f'Average particle cloud height t={t:0f} is\t{y_avg:.2E} m')
print(f'Average particle cloud width t={t:0f} is\t\t{x_avg:.2E} m')

