import numpy as np
from collections import namedtuple
from particle import ParticleCloud
from fields import  ElectricField

num_particles = 10_000

dynamic_viscosity_carrier = 0.001  # Pa*s
temperature = 293.15  # K
particle_diameter = 10E-9  # 10 nm
electrophoretic_mobility = -4.0E-8  # µm X cm / (V X s) E-6E-2 = E-8
dt = 0.5  # seconds
carrier_velocity = 1.0E-2 / 60.0  # m/s (1 ml/ min = 1cm^3/60 seconds, 1 cm /60 seconds)
e_field = ElectricField(magnitude=25) # V/m


channel_width = 3.2E-2  # cm
channel_height = 178E-6  # µm

particle_cloud = ParticleCloud(n_particles=num_particles,
                               initial_coordinate=(0, 0.75E-3, channel_height, channel_height),
                               particle_diameter=10E-9)

diff_coeff = particle_cloud.get_diffusion_coef(temp=temperature,
                                               dynamic_visc=dynamic_viscosity_carrier)

theoretical = diff_coeff / (e_field.get_component_y(em=electrophoretic_mobility))

print(f'Theoretical height: {theoretical:.2E}')

ColumnIO = namedtuple("ColumnIO", "x1 x2 y1 y2")

outlet_x = channel_width
outlet_y = (0, channel_height)


# calculate the initial average height
y_avg = particle_cloud.average_y()
x_avg = particle_cloud.average_x()
print(f'Average particle cloud height t=0 is\t{y_avg:.2E} m')
print(f'Average particle cloud width t=0 is\t\t{x_avg:.2E} m')


def apply_boundary_conditions(pc: ParticleCloud):
    # top
    particles_past_boundary = np.where(pc.y > channel_height)
    new_y = channel_height - (pc.y[particles_past_boundary] - channel_height)
    pc.y[particles_past_boundary] = new_y

    # bottom
    particles_past_boundary = np.where(pc.y < 0)
    new_y = -1.0 * pc.y[particles_past_boundary]
    pc.y[particles_past_boundary] = new_y

    #left
    particles_past_boundary = np.where(pc.x < 0)
    new_x = -1.0 * pc.x[particles_past_boundary]
    pc.x[particles_past_boundary] = new_x


def step(pc: ParticleCloud):
    # propagate x coordinates
    # x(t+dt) = x(t) + n * ld(dt) + vpx * dt
    n = np.random.normal(loc=0, scale=1, size=num_particles)
    ld = np.sqrt(2.0 * diff_coeff * dt)
    vpx = 6.0 * carrier_velocity * (
                np.divide(pc.y, channel_height) - np.square(np.divide(pc.y, channel_height)))
    pc.x += n * ld + vpx * dt
    # propagate y coordinates
    # y(t+dt) = y(t) + n * ld(dt) + vpy * dt
    y_field = e_field.get_component_y(em=electrophoretic_mobility) * dt
    pc.y += n * ld + y_field

t = 0
while t <= 300:
    step(particle_cloud)
    apply_boundary_conditions(particle_cloud)
    t += dt

y_avg = particle_cloud.average_y()
x_avg = particle_cloud.average_x()
print(f'Average particle cloud height t={t:0f} is\t{y_avg:.2E} m')
print(f'Average particle cloud width t={t:0f} is\t\t{x_avg:.2E} m')

