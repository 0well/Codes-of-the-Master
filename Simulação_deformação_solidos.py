import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Configuração inicial dos parâmetros
lambda1_initial = 2.0
lambda2_initial = 1.5
lambda3_initial = 0.8

# Definir os vértices do cubo unitário inicial
vertices = np.array([
    [0, 0, 0],  # O
    [1, 0, 0],  # A
    [1, 1, 0],  # B
    [0, 1, 0],  # C
    [0, 0, 1],  # D
    [1, 0, 1],  # E
    [1, 1, 1],  # F
    [0, 1, 1]   # G
])

# Definir as faces do cubo
faces = [
    [0, 1, 2, 3],  # Face OABC (inferior)
    [4, 5, 6, 7],  # Face superior
    [0, 1, 5, 4],  # Face frontal
    [2, 3, 7, 6],  # Face traseira
    [1, 2, 6, 5],  # Face direita
    [0, 3, 7, 4]   # Face esquerda
]

# Função para aplicar a deformação
def deform_cube(vertices, lambda1, lambda2, lambda3):
    deformed_vertices = np.zeros_like(vertices)
    for i, v in enumerate(vertices):
        X1, X2, X3 = v
        deformed_vertices[i] = [lambda1 * X1, -lambda3 * X3, lambda2 * X2]
    return deformed_vertices

# Função para calcular o tensor F
def get_F(lambda1, lambda2, lambda3):
    return np.array([
        [lambda1, 0, 0],
        [0, 0, -lambda3],
        [0, lambda2, 0]
    ])

# Função para calcular o tensor U
def get_U(lambda1, lambda2, lambda3):
    return np.array([
        [lambda1, 0, 0],
        [0, lambda2, 0],
        [0, 0, lambda3]
    ])

# Função para calcular o tensor de rotação R
def get_R(lambda1, lambda2, lambda3):
    F = get_F(lambda1, lambda2, lambda3)
    U = get_U(lambda1, lambda2, lambda3)
    U_inv = np.linalg.inv(U)
    return np.dot(F, U_inv)

# Configuração da figura
#fig = plt.figure(figsize=(16, 12)) (tamanho padrão )
fig = plt.figure(figsize=(8, 6), dpi=100) #(tamanho reduzido de 600x800)
fig.suptitle('Simulação de Deformação do Cubo Unitário', fontsize=16)

# Criar subplots para visualização 3D
ax1 = fig.add_subplot(231, projection='3d')
ax2 = fig.add_subplot(232, projection='3d')
ax3 = fig.add_subplot(233, projection='3d')

# Adicionar sliders para controlar os parâmetros
ax_lambda1 = plt.axes([0.25, 0.05, 0.65, 0.03])
ax_lambda2 = plt.axes([0.25, 0.01, 0.65, 0.03])
ax_lambda3 = plt.axes([0.25, 0.09, 0.65, 0.03])

slider_lambda1 = Slider(ax_lambda1, 'λ₁', 0.1, 3.0, valinit=lambda1_initial)
slider_lambda2 = Slider(ax_lambda2, 'λ₂', 0.1, 3.0, valinit=lambda2_initial)
slider_lambda3 = Slider(ax_lambda3, 'λ₃', 0.1, 3.0, valinit=lambda3_initial)

# Função para atualizar a visualização
def update(val):
    lambda1 = slider_lambda1.val
    lambda2 = slider_lambda2.val
    lambda3 = slider_lambda3.val
    
    # Limpar os subplots
    ax1.cla()
    ax2.cla()
    ax3.cla()
    
    # Aplicar deformação
    deformed_vertices = deform_cube(vertices, lambda1, lambda2, lambda3)
    
    # Plotar cubo original
    plot_cube(ax1, vertices, 'Cubo Unitário Original', 'blue')
    
    # Plotar cubo deformado
    plot_cube(ax2, deformed_vertices, 'Cubo Deformado', 'red')
    
    # Plotar rotação
    plot_rotation(ax3, lambda1, lambda2, lambda3)
    
    # Adicionar informações textuais
    detF = lambda1 * lambda2 * lambda3
    area_ratio = lambda1 * lambda2
    
    info_text = f'Det(F) = {detF:.2f}\nVolume Deformado: {detF:.2f}\nÁrea deformada/OABC = {area_ratio:.2f}'
    ax2.text2D(0.0, -0.4, info_text, transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    fig.canvas.draw_idle()

# Função para plotar um cubo
def plot_cube(ax, vertices, title, color):
    ax.set_title(title)
    
    # Plotar vértices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=color, s=50)
    
    # Plotar faces com preenchimento
    for i, face in enumerate(faces):
        verts = [vertices[vertex_idx] for vertex_idx in face]
        poly = Poly3DCollection([verts], alpha=0.2, linewidths=1, edgecolors='black')
        poly.set_facecolor(color)
        ax.add_collection3d(poly)
    
    # Destacar face OABC
    face_oabc = [vertices[i] for i in faces[0]]
    face_oabc.append(face_oabc[0])  # Fechar o polígono
    face_oabc = np.array(face_oabc)
    ax.plot(face_oabc[:, 0], face_oabc[:, 1], face_oabc[:, 2], color='green', linewidth=3)
    
    # Configurar limites dos eixos
    max_range = max(np.ptp(vertices[:, 0]), np.ptp(vertices[:, 1]), np.ptp(vertices[:, 2])) * 1.2
    mid_x = (np.max(vertices[:, 0]) + np.min(vertices[:, 0])) * 0.5
    mid_y = (np.max(vertices[:, 1]) + np.min(vertices[:, 1])) * 0.5
    mid_z = (np.max(vertices[:, 2]) + np.min(vertices[:, 2])) * 0.5
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    ax.set_xlabel('X/x₁')
    ax.set_ylabel('Y/x₂')
    ax.set_zlabel('Z/x₃')
    
    # Adicionar setas para os eixos
    ax.quiver(0, 0, 0, max_range/2, 0, 0, color='r', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, max_range/2, 0, color='g', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, max_range/2, color='b', arrow_length_ratio=0.1)
    
    ax.text(max_range/2, 0, 0, "e₁", color='r')
    ax.text(0, max_range/2, 0, "e₂", color='g')
    ax.text(0, 0, max_range/2, "e₃", color='b')

# Função para visualizar a rotação
def plot_rotation(ax, lambda1, lambda2, lambda3):
    ax.set_title('Tensor de Rotação R e Vetor Axial')
    
    # Calcular R
    R = get_R(lambda1, lambda2, lambda3)
    
    # Plotar os vetores transformados pelos eixos base
    origin = [0, 0, 0]
    
    # Eixos originais
    ax.quiver(*origin, 1, 0, 0, color='r', label='e₁', arrow_length_ratio=0.1)
    ax.quiver(*origin, 0, 1, 0, color='g', label='e₂', arrow_length_ratio=0.1)
    ax.quiver(*origin, 0, 0, 1, color='b', label='e₃', arrow_length_ratio=0.1)
    
    # Eixos após rotação
    e1_rotated = R.dot([1, 0, 0])
    e2_rotated = R.dot([0, 1, 0])
    e3_rotated = R.dot([0, 0, 1])
    
    ax.quiver(*origin, *e1_rotated, color='r', linestyle='--', arrow_length_ratio=0.1)
    ax.quiver(*origin, *e2_rotated, color='g', linestyle='--', arrow_length_ratio=0.1)
    ax.quiver(*origin, *e3_rotated, color='b', linestyle='--', arrow_length_ratio=0.1)
    
    # Plotar vetor axial (eixo de rotação)
    ax.quiver(*origin, 1.5, 0, 0, color='orange', linewidth=3, label='Vetor Axial (e₁)', arrow_length_ratio=0.1)
    
    # Configurar limites
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # LEGENDA NA POSIÇÃO ORIGINAL
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.6))
    
    # Adicionar informações sobre R
    info_text = f'R = {R}\nÂngulo de rotação: 90° em torno de e₁'
    ax.text2D(0.0, -0.5, info_text, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))

# Adicionar texto explicativo
plt.figtext(0.0, 0.3, 
            "Deformação: x₁ = λ₁X₁, x₂ = -λ₃X₃, x₃ = λ₂X₂\n\n"
            "(a) Volume deformado: ΔV = λ₁λ₂λ₃·ΔV₀\n"
            "(b) Área deformada de OABC: ΔA = λ₁λ₂ (com normal em e₂)\n"
            "(c) Tensor de rotação R = [[1,0,0],[0,0,-1],[0,1,0]]\n"
            "    Vetor axial: e₁ (rotação de 90° em torno de e₁)",
            bbox=dict(facecolor='white', alpha=0.7))


# Configurar layout
plt.tight_layout(rect=[0, 0.15, 1, 0.95])

# Inicializar a visualização
update(None)

# Conectar sliders à função de atualização
slider_lambda1.on_changed(update)
slider_lambda2.on_changed(update)
slider_lambda3.on_changed(update)

plt.show()