import streamlit as st
import numpy as np
import plotly.graph_objects as go
from hydrogen_sim import WaveFunction as wf

st.title("Hydrogen Electron Cloud Simulation")
st.sidebar.header("Set Quantum Numbers")

n = st.sidebar.number_input("Principal quantum number (n)", min_value=1, max_value=10, value=1, step=1)
l = st.sidebar.number_input("Azimuthal quantum number (l)", min_value=0, max_value=n-1, value=0, step=1, key=f"l_for_n{n}")
m = st.sidebar.number_input("Magnetic quantum number (m)", min_value=-l, max_value=l, value=0, step=1, key=f"m_for_l{l}")
space = st.sidebar.slider("Coordinate Size", min_value=10, max_value=50, value=15, step=5)
resolution = st.sidebar.slider("Coordinate Resolution", min_value=10, max_value=100, value=50, step=2)

if st.sidebar.button("Start Simulation"):
    with st.spinner("Calculating probability density..."):
        x = np.linspace(-space, space, resolution)
        y = np.linspace(-space, space, resolution)
        z = np.linspace(-space, space, resolution)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        r = np.sqrt(X**2 + Y**2 + Z**2)
        theta = np.arccos(Z / r)
        phi = np.arctan2(Y, X)
        
        a = 1
        P = wf.ProbabilityDensity(r, theta, phi, a, n, l, m)
        
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()
        P_flat = P.flatten()
        
        random_thresholds = np.random.rand(len(P_flat)) * np.max(P_flat)
        mask = P_flat > random_thresholds
        
        x_plot = X_flat[mask]
        y_plot = Y_flat[mask]
        z_plot = Z_flat[mask]
        color_density = P_flat[mask]
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x_plot,
            y=y_plot,
            z=z_plot,
            mode='markers',
            marker=dict(
                size=2,              
                color=color_density, 
                colorscale='magma',  
                opacity=0.5          
            )
        )])
        
        fig.update_layout(
            title=f'Probability Density for Psi_{n}{l}{m}',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False)
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)