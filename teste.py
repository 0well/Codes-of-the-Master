import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

plt.style.use('ggplot')

# --- Definições iniciais ---

# Vértices do cubo unitário
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

# Faces do cubo (índices dos vértices)
faces = [
    [0, 1, 2, 3],  # Inferior OABC
    [4, 5, 6, 7],  # Superior DEFG
    [0, 1, 5, 4],  # Frontal OAE D
    [2, 3, 7, 6],  # Traseira BCGF
    [1, 2, 6, 5],  # Direita ABFE
    [0, 3, 7, 4]   # Esquerda OCGD
]

# --- Funções ---

def deform_cube(vertices, λ1, λ2, λ3):
    """
    Aplica a deformação ao cubo segundo:
    x1 = λ1 * X1
    x2 = -λ3 * X3
    x3 = λ2 * X2
    """
    X1, X2, X3 = vertices[:,0], vertices[:,1], vertices[:,2]
    x1 = λ1 * X1
    x2 = -λ3 * X3
    x3 = λ2 * X2
    return np.column_stack((x1, x2, x3))

def get_F(λ1, λ2, λ3):
    """Tensor de deformação F"""
    return np.array([
        [λ1, 0, 0],
        [0, 0, -λ3],
        [0, λ2, 0]
    ])

def get_U(λ1, λ2, λ3):
    """Tensor U"""
    return np.diag([λ1, λ2, λ3])

def get_R(λ1, λ2, λ3):
    """Tensor de rotação R = F * U^{-1}"""
    F = get_F(λ1, λ2, λ3)
    U = get_U(λ1, λ2, λ3)
    U_inv = np.linalg.inv(U)
    return F @ U_inv

def plot_cube(ax, verts, color, alpha=0.3, edge_color='k', title=None):
    """Plota o cubo com faces transparentes e arestas"""
    for face in faces:
        square = verts[face]
        # Fechar o polígono
        square = np.vstack([square, square[0]])
        ax.plot(square[:,0], square[:,1], square[:,2], color=edge_color, lw=1.5)
        ax.add_collection3d(plt.Poly3DCollection([square], facecolors=color, linewidths=0.5, alpha=alpha))
    ax.scatter(verts[:,0], verts[:,1], verts[:,2], color=edge_color, s=30)
    if title:
        ax.set_title(title, fontsize=12, pad=10)
    set_axes_equal(ax)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')

def set_axes_equal(ax):
    """Define escala igual para os eixos 3D"""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    spans = limits[:,1] - limits[:,0]
    max_span = max(spans)
    centers = np.mean(limits, axis=1)
    ax.set_xlim3d(centers[0] - max_span/2, centers[0] + max_span/2)
    ax.set_ylim3d(centers[1] - max_span/2, centers[1] + max_span/2)
    ax.set_zlim3d(centers[2] - max_span/2, centers[2] + max_span/2)

def plot_rotation(ax, R):
    """Plota os vetores base originais e rotacionados"""
    origin = np.zeros(3)
    colors = ['r', 'g', 'b']
    labels = ['$e_1$', '$e_2$', '$e_3$']

    # Vetores originais
    for i, c in enumerate(colors):
        vec = np.zeros(3)
        vec[i] = 1
        ax.quiver(*origin, *vec, color=c, length=1, normalize=True, label=f'Original {labels[i]}')

    # Vetores rotacionados
    for i, c in enumerate(colors):
        vec = R[:,i]
        ax.quiver(*origin, *vec, color=c, length=1, normalize=True, linestyle='dashed', label=f'Rotacionado {labels[i]}')

    ax.set_title('Tensor de Rotação $R$')
    set_axes_equal(ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper left', fontsize=8)

# --- Configuração da figura e sliders ---

fig = plt.figure(figsize=(16, 6))
fig.suptitle('Deformação do Cubo Unitário com Controle Interativo', fontsize=16, y=1.05)

# Subplots
ax_orig = fig.add_subplot(131, projection='3d')
ax_def = fig.add_subplot(132, projection='3d')
ax_rot = fig.add_subplot(133, projection='3d')

# Sliders
slider_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03], facecolor='lightgray')
slider_ax2 = fig.add_axes([0.15, 0.01, 0.7, 0.03], facecolor='lightgray')
slider_ax3 = fig.add_axes([0.15, 0.09, 0.7, 0.03], facecolor='lightgray')

slider_λ1 = Slider(slider_ax, r'$\lambda_1$', 0.1, 3.0, valinit=2.0)
slider_λ2 = Slider(slider_ax2, r'$\lambda_2$', 0.1, 3.0, valinit=1.5)
slider_λ3 = Slider(slider_ax3, r'$\lambda_3$', 0.1, 3.0, valinit=0.8)

def update(val):
    λ1 = slider_λ1.val
    λ2 = slider_λ2.val
    λ3 = slider_λ3.val

    ax_orig.cla()
    ax_def.cla()
    ax_rot.cla()

    # Cubo original
    plot_cube(ax_orig, vertices, color='skyblue', alpha=0.2, edge_color='blue', title='Cubo Original')

    # Cubo deformado
    verts_def = deform_cube(vertices, λ1, λ2, λ3)
    plot_cube(ax_def, verts_def, color='salmon', alpha=0.4, edge_color='red', title='Cubo Deformado')

    # Tensor de rotação
    R = get_R(λ1, λ2, λ3)
    plot_rotation(ax_rot, R)

    # Informações
    detF = np.linalg.det(get_F(λ1, λ2, λ3))
    area_ratio = λ1 * λ2
    info = f'Det(F) = {detF:.3f}\nÁrea deformada OABC = {area_ratio:.3f}'
    ax_def.text2D(0.05, -0.15, info, transform=ax_def.transAxes,
                  bbox=dict(facecolor='white', alpha=0.7), fontsize=10)

    fig.canvas.draw_idle()

update(None)

slider_λ1.on_changed(update)
slider_λ2.on_changed(update)
slider_λ3.on_changed(update)

plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0.3)
plt.show()


'''
''' import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Parâmetros de deformação
λ1, λ2, λ3 = 1.2, 1.5, 0.8  # pode mudar os valores

# Tensor gradiente de deformação F
F = np.array([
    [λ1, 0, 0],
    [0, 0, -λ3],
    [0, λ2, 0]
])

# Definição do cubo unitário (vértices)
vertices = np.array([
    [0,0,0],
    [1,0,0],
    [1,1,0],
    [0,1,0],
    [0,0,1],
    [1,0,1],
    [1,1,1],
    [0,1,1]
])

# Aplica a deformação: x = F*X
vertices_def = vertices @ F.T

# Faces do cubo (índices dos vértices)
faces = [
    [0,1,2,3],
    [4,5,6,7],
    [0,1,5,4],
    [2,3,7,6],
    [1,2,6,5],
    [0,3,7,4]
]

# Função para plotar cubo
def plot_cube(ax, verts, faces, color, alpha=0.3):
    for face in faces:
        poly = Poly3DCollection([[verts[i] for i in face]], alpha=alpha, facecolor=color)
        ax.add_collection3d(poly)
    ax.scatter(verts[:,0], verts[:,1], verts[:,2], c=color, s=40)

# Plot
fig = plt.figure(figsize=(10,5))

# Cubo original
ax1 = fig.add_subplot(121, projection='3d')
plot_cube(ax1, vertices, faces, 'blue', 0.2)
ax1.set_title("Cubo Original")
ax1.set_xlabel('X1'); ax1.set_ylabel('X2'); ax1.set_zlabel('X3')
ax1.set_xlim([0,2]); ax1.set_ylim([0,2]); ax1.set_zlim([0,2])

# Cubo deformado
ax2 = fig.add_subplot(122, projection='3d')
plot_cube(ax2, vertices_def, faces, 'red', 0.2)
ax2.set_title("Cubo Deformado")
ax2.set_xlabel('x1'); ax2.set_ylabel('x2'); ax2.set_zlabel('x3')
ax2.set_xlim([0,2]); ax2.set_ylim([-2,2]); ax2.set_zlim([-2,2])

plt.show()
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

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
fig = plt.figure(figsize=(16, 12))
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
    
    info_text = f'Det(F) = {detF:.2f}\nÁrea deformada/OABC = {area_ratio:.2f}'
    ax2.text2D(0.0, -0.4, info_text, transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    fig.canvas.draw_idle()

# Função para plotar um cubo
def plot_cube(ax, vertices, title, color):
    ax.set_title(title)
    
    # Plotar vértices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=color, s=50)
    
    # Plotar arestas
    for face in faces:
        for i in range(4):
            start = vertices[face[i]]
            end = vertices[face[(i+1)%4]]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color, alpha=0.6)
    
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

plt.show()'''