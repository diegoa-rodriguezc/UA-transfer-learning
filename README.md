# Transfer Learning

Implementaciones de **Transfer Learning (TL)** en **Deep Reinforcement Learning (RL)**, sobre dos entornos clásicos de [Gymnasium](https://gymnasium.farama.org) (CartPole y HalfCheetah).

## ¿Qué es Transfer Learning en Deep RL?

En Reinforcement Learning un agente aprende una política $\pi(a|s)$ interactuando con el entorno mediante prueba y error. Esto suele requerir millones de interacciones y exploración intensiva, lo que en problemas reales es lento o costoso.

**Transfer Learning** consiste en aprovechar conocimiento ya adquirido en una **tarea fuente** (*source*) para acelerar el aprendizaje en una **tarea objetivo** (*target*). El conocimiento transferido puede tomar muchas formas: una función de valor, una política experta, demostraciones, una representación intermedia del estado o una recompensa auxiliar.

El survey (paper) de Zhu et al. organiza estos métodos en torno a tres preguntas:

1. **¿Qué se transfiere?** — recompensa, política, valor, representación, demostraciones.
2. **¿Cómo se transfiere?** — *fine-tuning*, distillation, *reward shaping*, mapeo entre dominios, etc.
3. **¿Cuándo es útil?** — se mide con métricas como *Jumpstart* (jp), *Asymptotic performance* (ap), *Time to Threshold* (tt) y *Accumulated reward* (ar).

Los dos notebooks de este repositorio implementan métodos representativos de estas categorías y los comparan contra una línea base sin transferencia, usando exactamente esas métricas.


## Contenido del repositorio

```text
UA-transfer-learning/
├── 001_TL_RL_Notebook.ipynb              # DQN + Policy Distillation + PBRS (CartPole-v1)
├── 001_TL_RL_Notebook.html               # Export HTML del notebook 001
├── 002_TL_Representation_Transfer.ipynb  # TD3 + Frozen / Fine-Tuning (HalfCheetah-v5)
├── 002_TL_Representation_Transfer.html   # Export HTML del notebook 002
├── transfer_learning.pdf                 # Survey(paper) de Zhu, Lin & Zhou (2023)
├── TL_DRL_Presentacion.pdf               # Presentación realizada sobre Transfer Learning
├── README.md
└── .gitignore
```

### Notebooks

* [`001_TL_RL_Notebook.ipynb`](001_TL_RL_Notebook.ipynb) — TL en acciones discretas (DQN sobre CartPole-v1)

Implementa tres experimentos comparativos sobre **CartPole-v1**:

| Experimento | Categoría en el survey | Idea central |
|---|---|---|
| **DQN Baseline** | RL sin TL | DQN clásico entrenado desde cero (300 episodios). |
| **Policy Distillation** | Policy Transfer (§4.3.1) | Loss híbrida `L = (1-α)·L_RL + α·L_distill`, donde `L_distill = KL( softmax(Q_teacher/τ) ‖ softmax(Q_student/τ) )` (Rusu et al., 2015; Hinton et al., 2014). |
| **Potential-Based Reward Shaping (PBRS)** | Reward Shaping (§4.1) | Recompensa modificada `R'(s,a,s') = R + γ·Φ(s') − Φ(s)` con potencial `Φ(s) = V_teacher(s) = max_a Q_teacher(s,a)`. Preserva la política óptima (Ng, Harada & Russell, 1999). |

**Métricas finales reportadas en el notebook** (asymptotic, promedio de los últimos 30 episodios):

| Método | jp | ap | ar | Transfer Ratio |
|---|---:|---:|---:|---:|
| DQN Baseline | 18.8 | 122.8 | 16,878 | — |
| Policy Distillation | 60.0 | 60.5 | 19,025 | 0.49 |
| PBRS Reward Shaping | 21.9 | 140.4 | 17,384 | **1.14** |

El notebook incluye además visualizaciones de las distribuciones de acción *teacher* vs *student* y de la función potencial Φ(s) a lo largo de una trayectoria.

* [`002_TL_Representation_Transfer.ipynb`](002_TL_Representation_Transfer.ipynb) — TL en control continuo (TD3 sobre HalfCheetah-v5)

Implementa tres experimentos comparativos sobre **HalfCheetah-v5** (MuJoCo) con el algoritmo **TD3** (Fujimoto, van Hoof & Meger, 2018):

| Experimento | Categoría en el survey (paper) | Idea central |
|---|---|---|
| **TD3 Baseline** | RL sin TL | TD3 entrenado desde cero con presupuesto reducido (50k *timesteps*). |
| **Source Agent** | — | TD3 entrenado con presupuesto amplio (100k *timesteps*); sirve como fuente de representaciones. |
| **Frozen Transfer** | Representation Transfer (§4.2) | El *feature extractor* (capas inferiores) se copia del *source* y se **congela**; sólo se entrenan las *heads*. |
| **Fine-Tuning Transfer** | Representation Transfer (§4.2) | Todos los pesos se inicializan desde el *source*; todas las capas son entrenables con **learning rates diferenciados** (LR bajo para el *feature extractor*, LR normal para las *heads*). |

Tanto `Actor` como `Critic` están diseñados con una separación explícita `feature_extractor` / `head` para hacer la transferencia natural a nivel de código.

**Métricas finales reportadas en el notebook** (asymptotic, promedio de los últimos 20 episodios):

| Método | jp | ap | ar | Transfer Ratio |
|---|---:|---:|---:|---:|
| TD3 Baseline (50k) | −393.8 | 589.6 | 9,026 | — |
| Frozen Transfer | −175.3 | −173.4 | −7,979 | −0.29 |
| Fine-Tuning Transfer | 573.4 | **999.8** | **42,747** | **1.70** |

El notebook también genera videos `.mp4` de la locomoción aprendida por cada agente y un análisis de las representaciones (PCA 2D + similitud coseno) que muestra qué tanto se parecen las activaciones del *feature extractor* entre los distintos agentes.

### Archivos adicionales

| Archivo | Descripción |
|---|---|
| [`transfer_learning.pdf`](transfer_learning.pdf) | Survey de Zhu, Lin & Zhou (2023) — referencia teórica del proyecto. Las secciones citadas en los notebooks (§4.1, §4.2, §4.3.1) corresponden a este documento. |
| [`001_TL_RL_Notebook.html`](001_TL_RL_Notebook.html) | Export HTML del primer notebook con todas las salidas y gráficas. |
| [`002_TL_Representation_Transfer.html`](002_TL_Representation_Transfer.html) | Export HTML del segundo notebook con todas las salidas y gráficas. |
| [`TL_DRL_Presentacion.pdf`](TL_DRL_Presentacion.pdf) | Presentación sobre Transfer Learning. |
| [Vídeo de presentación](https://uniandes-my.sharepoint.com/:v:/g/personal/da_rodriguezc123_uniandes_edu_co/IQCxdL-1TZ7uQIP4JLu0SWv8AUwkif9TaS9_2aCXOnkSPZA?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=Pr293T) | Vídeo presentación sobre Transfer Learning. *(Para acceder al vídeo se debe tener un dominio @uniandes.edu.co)* |


## Referencias

1. **Zhu, Lin & Zhou (2023).** *Transfer Learning in Deep RL: A Survey.* IEEE TPAMI 45(11). — referencia central del proyecto.
2. **Ng, Harada & Russell (1999).** *Policy invariance under reward transformations.* ICML. — fundamento teórico de PBRS.
3. **Hinton, Vinyals & Dean (2014).** *Distilling the Knowledge in a Neural Network.* NIPS Workshop.
4. **Rusu et al. (2015).** *Policy Distillation.* arXiv:1511.06295.
5. **Fujimoto, van Hoof & Meger (2018).** *Addressing Function Approximation Error in Actor-Critic Methods (TD3).* ICML.
6. **Parisotto, Ba & Salakhutdinov (2016).** *Actor-Mimic: Deep Multitask and Transfer RL.* ICLR.
7. **Rusu et al. (2016).** *Progressive Neural Networks.* arXiv:1606.04671.
