#!/usr/bin/env python3
"""
Visualize pointcloud files (.pcd or .npy) and coordinate systems represented by 4x4 matrices
Usage: python vis_pcd.py <pcd_file> [matrix_file] [--port PORT] [--host HOST]
"""

import argparse
import numpy as np
import open3d as o3d
from flask import Flask, render_template_string
import plotly.graph_objs as go
import plotly.io as pio
from pathlib import Path
import json
import sys
import os

class PointcloudVisualizer:
    def __init__(self, host='0.0.0.0', port=5000):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        
    def load_pointcloud(self, pcd_file):
        """Load pointcloud from .pcd or .npy file"""
        pcd_path = Path(pcd_file)
        
        if not pcd_path.exists():
            raise FileNotFoundError(f"Pointcloud file not found: {pcd_file}")
            
        if pcd_path.suffix.lower() == '.pcd':
            # Load .pcd file using Open3D
            pcd = o3d.io.read_point_cloud(str(pcd_path))
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None
            
            if colors is not None:
                # Combine points and colors
                pointcloud = np.hstack([points, colors * 255])  # Scale colors to 0-255
            else:
                pointcloud = points
                
        elif pcd_path.suffix.lower() == '.npy':
            # Load .npy file
            pointcloud = np.load(pcd_path)
            pointcloud[:, 3:6] *= 255
            
        else:
            raise ValueError(f"Unsupported file format: {pcd_path.suffix}")
            
        print(f"Loaded pointcloud with shape: {pointcloud.shape}")
        return pointcloud
        
    def load_matrices(self, matrix_file):
        """Load 4x4 matrices from file"""
        if matrix_file is None:
            return []
            
        matrix_path = Path(matrix_file)
        if not matrix_path.exists():
            print(f"Warning: Matrix file not found: {matrix_file}")
            return []
            
        try:
            if matrix_path.suffix.lower() == '.npy':
                matrices = np.load(matrix_path)
                # Ensure it's a list of 4x4 matrices
                if matrices.ndim == 2 and matrices.shape == (4, 4):
                    matrices = [matrices]
                elif matrices.ndim == 3 and matrices.shape[1:] == (4, 4):
                    matrices = list(matrices)
                else:
                    raise ValueError(f"Invalid matrix shape: {matrices.shape}")
                    
            elif matrix_path.suffix.lower() == '.json':
                with open(matrix_path, 'r') as f:
                    data = json.load(f)
                matrices = [np.array(m) for m in data]
                
            elif matrix_path.suffix.lower() == '.txt':
                matrices = []
                with open(matrix_path, 'r') as f:
                    lines = f.readlines()
                    
                current_matrix = []
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    try:
                        row = [float(x) for x in line.split()]
                        if len(row) == 4:
                            current_matrix.append(row)
                            if len(current_matrix) == 4:
                                matrices.append(np.array(current_matrix))
                                current_matrix = []
                    except ValueError:
                        continue
                        
            else:
                raise ValueError(f"Unsupported matrix file format: {matrix_path.suffix}")
                
            print(f"Loaded {len(matrices)} matrices")
            return matrices
            
        except Exception as e:
            print(f"Error loading matrices: {e}")
            return []
    
    def create_coordinate_frame_traces(self, matrices, scale=0.1):
        """Create traces for coordinate frames"""
        traces = []
        
        for i, matrix in enumerate(matrices):
            # Extract origin and axes from 4x4 matrix
            origin = matrix[:3, 3]
            x_axis = matrix[:3, 0] * scale
            y_axis = matrix[:3, 1] * scale
            z_axis = matrix[:3, 2] * scale
            
            # X axis (red)
            traces.append(go.Scatter3d(
                x=[origin[0], origin[0] + x_axis[0]],
                y=[origin[1], origin[1] + x_axis[1]],
                z=[origin[2], origin[2] + x_axis[2]],
                mode='lines+markers',
                line=dict(color='red', width=8),
                marker=dict(size=[3, 0]),  # Only show marker at origin, no marker at end
                name=f'Frame {i} X-axis',
                showlegend=True
            ))
            
            # Y axis (green)
            traces.append(go.Scatter3d(
                x=[origin[0], origin[0] + y_axis[0]],
                y=[origin[1], origin[1] + y_axis[1]],
                z=[origin[2], origin[2] + y_axis[2]],
                mode='lines+markers',
                line=dict(color='green', width=8),
                marker=dict(size=[3, 0]),  # Only show marker at origin, no marker at end
                name=f'Frame {i} Y-axis',
                showlegend=True
            ))
            
            # Z axis (blue)
            traces.append(go.Scatter3d(
                x=[origin[0], origin[0] + z_axis[0]],
                y=[origin[1], origin[1] + z_axis[1]],
                z=[origin[2], origin[2] + z_axis[2]],
                mode='lines+markers',
                line=dict(color='blue', width=8),
                marker=dict(size=[3, 0]),  # Only show marker at origin, no marker at end
                name=f'Frame {i} Z-axis',
                showlegend=True
            ))
            
        return traces
    
    def create_pointcloud_trace(self, pointcloud, color=None, size=3, opacity=0.7):
        """Create trace for pointcloud"""
        x_coords = pointcloud[:, 0]
        y_coords = pointcloud[:, 1]
        z_coords = pointcloud[:, 2]

        if pointcloud.shape[1] == 3:
            if color is None:
                # Design a colorful point cloud based on 3d coordinates
                min_coords = pointcloud.min(axis=0)
                max_coords = pointcloud.max(axis=0)
                normalized_coords = (pointcloud - min_coords) / (max_coords - min_coords)
                try:
                    colors = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in normalized_coords]
                except:
                    colors = ['rgb(0,255,255)' for _ in range(len(x_coords))]
            else:    
                colors = ['rgb({},{},{})'.format(color[0], color[1], color[2]) for _ in range(len(x_coords))]
        else:
            # Use existing colors
            colors = ['rgb({},{},{})'.format(int(r), int(g), int(b)) for r, g, b in pointcloud[:, 3:6]]

        return go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=size,
                opacity=opacity,
                color=colors
            ),
            name='Pointcloud',
            showlegend=True
        )
    
    def visualize(self, pointcloud, matrices, axis_scale=0.1):
        """Create visualization with pointcloud and coordinate frames"""
        traces = []
        
        # Add pointcloud trace
        pcd_trace = self.create_pointcloud_trace(pointcloud, size=2, opacity=0.8)
        traces.append(pcd_trace)
        
        # Add coordinate frame traces
        frame_traces = self.create_coordinate_frame_traces(matrices, scale=axis_scale)
        traces.extend(frame_traces)
        
        # Create layout
        layout = go.Layout(
            title="Pointcloud and Coordinate Frames Visualization",
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(
                xaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    gridcolor='lightgrey',
                    title='X'
                ),
                yaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    gridcolor='lightgrey',
                    title='Y'
                ),
                zaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    gridcolor='lightgrey',
                    title='Z'
                ),
                bgcolor='white',
                aspectmode='data'
            )
        )
        
        fig = go.Figure(data=traces, layout=layout)
        
        # Convert to HTML
        div = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        
        # Setup Flask route
        @self.app.route('/')
        def index():
            return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Pointcloud Visualization</title>
                <style>
                    body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
                    h1 { color: #333; }
                    .info { background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }
                </style>
            </head>
            <body>
                <h1>Pointcloud and Coordinate Frames Visualization</h1>
                <div class="info">
                    <strong>Pointcloud shape:</strong> {{ pcd_shape }}<br>
                    <strong>Number of coordinate frames:</strong> {{ num_frames }}<br>
                    <strong>Server:</strong> http://{{ host }}:{{ port }}
                </div>
                <div>{{ div|safe }}</div>
            </body>
            </html>
            ''', 
            div=div, 
            pcd_shape=pointcloud.shape,
            num_frames=len(matrices),
            host=self.host,
            port=self.port
            )
        
        print(f"Starting visualization server at http://{self.host}:{self.port}")
        print("Press Ctrl+C to stop the server")
        
        try:
            self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
        except KeyboardInterrupt:
            print("\nServer stopped.")

def main():
    parser = argparse.ArgumentParser(description='Visualize pointcloud and coordinate frames')
    parser.add_argument('pcd_file', help='Path to pointcloud file (.pcd or .npy)')
    parser.add_argument('matrix_file', nargs='?', default=None, 
                       help='Path to matrix file (.npy, .json, or .txt) containing 4x4 transformation matrices')
    parser.add_argument('--port', type=int, default=5000, help='Port for web server (default: 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for web server (default: 0.0.0.0)')
    parser.add_argument('--axis-scale', type=float, default=0.1, help='Scale factor for coordinate axes (default: 0.1)')
    
    args = parser.parse_args()
    
    try:
        # Create visualizer
        visualizer = PointcloudVisualizer(host=args.host, port=args.port)
        
        # Load pointcloud
        pointcloud = visualizer.load_pointcloud(args.pcd_file)
        
        # Load matrices
        matrices = visualizer.load_matrices(args.matrix_file)
        
        # Start visualization
        visualizer.visualize(pointcloud, matrices, axis_scale=args.axis_scale)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
