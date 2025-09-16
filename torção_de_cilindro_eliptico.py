import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

# Configuração inicial
fig = plt.figure(figsize=(15, 10))
plt.subplots_adjust(bottom=0.25)

# Parâmetros iniciais
a_init = 1.0  # semi-eixo menor
b_init = 2.0  # semi-eixo maior
Mt_init = 5.0  # torque aplicado
L = 5.0  # comprimento da barra

# Função para calcular as componentes da tensão de cisalhamento
def shear_stress_components(x2, x3, a, b, Mt):
    T12 = -(2 * Mt) / (np.pi * a * b**3) * x3
    T13 = (2 * Mt) / (np.pi * a**3 * b) * x2
    return T12, T13

# Função para calcular a magnitude da tensão de cisalhamento
def shear_stress_magnitude(x2, x3, a, b, Mt):
    T12, T13 = shear_stress_components(x2, x3, a, b, Mt)
    return np.sqrt(T12**2 + T13**2)

# Criar a superfície do cilindro elíptico
def create_elliptical_cylinder(a, b, length, num_points=50):
    theta = np.linspace(0, 2*np.pi, num_points)
    z = np.linspace(-length/2, length/2, num_points)
    theta, z = np.meshgrid(theta, z)
    
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    
    return x, y, z

# Criar a malha para a seção transversal
def create_cross_section(a, b, num_points=50):
    theta = np.linspace(0, 2*np.pi, num_points)
    r = np.linspace(0, 1, num_points)
    theta, r = np.meshgrid(theta, r)
    
    x2 = a * r * np.cos(theta)
    x3 = b * r * np.sin(theta)
    
    return x2, x3

# Primeiro subplot: Cilindro elíptico sem torção
ax1 = fig.add_subplot(131, projection='3d')
x1, y1, z1 = create_elliptical_cylinder(a_init, b_init, L)
surf1 = ax1.plot_surface(z1, x1, y1, alpha=0.8, color='lightblue')
ax1.set_title('Cilindro Elíptico sem Torção')
ax1.set_xlabel('Eixo X1')
ax1.set_ylabel('Eixo X2')
ax1.set_zlabel('Eixo X3')
ax1.set_xlim(-L/2, L/2)
ax1.set_ylim(-b_init*1.2, b_init*1.2)
ax1.set_zlim(-a_init*1.2, a_init*1.2)

# Segundo subplot: Cilindro elíptico com torção
ax2 = fig.add_subplot(132, projection='3d')
x2_plot, y2_plot, z2_plot = create_elliptical_cylinder(a_init, b_init, L)

# Aplicar torção (deformação por torção)
def apply_torsion(x, y, z, a, b, Mt, length):
    # Ângulo de torção por unidade de comprimento
    alpha = Mt * (a**2 + b**2) / (np.pi * a**3 * b**3)
    
    # Aplicar rotação proporcional à posição ao longo do eixo
    rotation_angle = alpha * z
    
    # Rotacionar cada ponto
    x_rotated = x * np.cos(rotation_angle) - y * np.sin(rotation_angle)
    y_rotated = x * np.sin(rotation_angle) + y * np.cos(rotation_angle)
    
    return x_rotated, y_rotated, z

x2_torsion, y2_torsion, z2_torsion = apply_torsion(x2_plot, y2_plot, z2_plot, a_init, b_init, Mt_init, L)
surf2 = ax2.plot_surface(z2_torsion, x2_torsion, y2_torsion, alpha=0.8, color='lightcoral')
ax2.set_title('Cilindro Elíptico com Torção')
ax2.set_xlabel('Eixo X1')
ax2.set_ylabel('Eixo X2')
ax2.set_zlabel('Eixo X3')
ax2.set_xlim(-L/2, L/2)
ax2.set_ylim(-b_init*1.2, b_init*1.2)
ax2.set_zlim(-a_init*1.2, a_init*1.2)

# Terceiro subplot: Vetores de tensão na seção transversal
ax3 = fig.add_subplot(133)
x2, x3 = create_cross_section(a_init, b_init, 20)

# Calcular componentes da tensão
T12, T13 = shear_stress_components(x2, x3, a_init, b_init, Mt_init)
T_mag = shear_stress_magnitude(x2, x3, a_init, b_init, Mt_init)

# Normalizar vetores para melhor visualização
norm_factor = np.max(T_mag) if np.max(T_mag) > 0 else 1
T12_norm = T12 / norm_factor
T13_norm = T13 / norm_factor

# Plotar vetores de tensão
quiver = ax3.quiver(x2, x3, T12_norm, T13_norm, T_mag, 
                   scale=20, cmap='viridis', width=0.005)
ax3.set_title('Vetores de Tensão na Seção Transversal')
ax3.set_xlabel('Eixo X2')
ax3.set_ylabel('Eixo X3')
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)

# Adicionar contorno da elipse
ellipse = Ellipse((0, 0), width=2*a_init, height=2*b_init, 
                  edgecolor='red', facecolor='none', linestyle='--')
ax3.add_patch(ellipse)

# Adicionar barra de cores
cbar = plt.colorbar(quiver, ax=ax3)
cbar.set_label('Magnitude da Tensão')

# Ajustar layout
plt.tight_layout(rect=[0, 0.25, 1, 0.95])

# Adicionar sliders para interação
ax_slider_a = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_slider_b = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_slider_Mt = plt.axes([0.25, 0.05, 0.65, 0.03])

slider_a = Slider(ax_slider_a, 'Semi-eixo menor (a)', 0.5, 3.0, valinit=a_init)
slider_b = Slider(ax_slider_b, 'Semi-eixo maior (b)', 0.5, 3.0, valinit=b_init)
slider_Mt = Slider(ax_slider_Mt, 'Torque (Mt)', 0.0, 10.0, valinit=Mt_init)

# Função de atualização
def update(val):
    a = slider_a.val
    b = slider_b.val
    Mt = slider_Mt.val
    
    # Atualizar primeiro subplot (sem torção)
    ax1.clear()
    x1, y1, z1 = create_elliptical_cylinder(a, b, L)
    surf1 = ax1.plot_surface(z1, x1, y1, alpha=0.8, color='lightblue')
    ax1.set_title('Cilindro Elíptico sem Torção')
    ax1.set_xlabel('Eixo X1')
    ax1.set_ylabel('Eixo X2')
    ax1.set_zlabel('Eixo X3')
    ax1.set_xlim(-L/2, L/2)
    ax1.set_ylim(-b*1.2, b*1.2)
    ax1.set_zlim(-a*1.2, a*1.2)
    
    # Atualizar segundo subplot (com torção)
    ax2.clear()
    x2_plot, y2_plot, z2_plot = create_elliptical_cylinder(a, b, L)
    x2_torsion, y2_torsion, z2_torsion = apply_torsion(x2_plot, y2_plot, z2_plot, a, b, Mt, L)
    surf2 = ax2.plot_surface(z2_torsion, x2_torsion, y2_torsion, alpha=0.8, color='lightcoral')
    ax2.set_title('Cilindro Elíptico com Torção')
    ax2.set_xlabel('Eixo X1')
    ax2.set_ylabel('Eixo X2')
    ax2.set_zlabel('Eixo X3')
    ax2.set_xlim(-L/2, L/2)
    ax2.set_ylim(-b*1.2, b*1.2)
    ax2.set_zlim(-a*1.2, a*1.2)
    
    # Atualizar terceiro subplot (vetores de tensão)
    ax3.clear()
    x2, x3 = create_cross_section(a, b, 20)
    
    T12, T13 = shear_stress_components(x2, x3, a, b, Mt)
    T_mag = shear_stress_magnitude(x2, x3, a, b, Mt)
    
    norm_factor = np.max(T_mag) if np.max(T_mag) > 0 else 1
    T12_norm = T12 / norm_factor
    T13_norm = T13 / norm_factor
    
    quiver = ax3.quiver(x2, x3, T12_norm, T13_norm, T_mag, 
                       scale=20, cmap='viridis', width=0.005)
    ax3.set_title('Vetores de Tensão na Seção Transversal')
    ax3.set_xlabel('Eixo X2')
    ax3.set_ylabel('Eixo X3')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    ellipse = Ellipse((0, 0), width=2*a, height=2*b, 
                      edgecolor='red', facecolor='none', linestyle='--')
    ax3.add_patch(ellipse)
    
    fig.canvas.draw_idle()

# Conectar sliders à função de atualização
slider_a.on_changed(update)
slider_b.on_changed(update)
slider_Mt.on_changed(update)

# Adicionar botão de reset
reset_ax = plt.axes([0.8, 0.01, 0.1, 0.04])
button_reset = Button(reset_ax, 'Reset', hovercolor='0.975')

def reset(event):
    slider_a.reset()
    slider_b.reset()
    slider_Mt.reset()

button_reset.on_clicked(reset)

plt.show()